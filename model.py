import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GRAPH ATTENTION LAYER
# ============================================================

class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_features=1,
        out_features=16,
        dropout=0.1
    ):
        super().__init__()

        self.out_features = out_features

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        self.attn_src = nn.Linear(
            out_features,
            1,
            bias=False
        )

        self.attn_dst = nn.Linear(
            out_features,
            1,
            bias=False
        )

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(
            out_features
        )

    def forward(self, x):

        # ----------------------------------------------------
        # x shape:
        #
        # [batch, nodes, features]
        #
        # For traffic speed:
        # features = 1
        # ----------------------------------------------------

        h = self.linear(x)

        # ----------------------------------------------------
        # Attention score
        # ----------------------------------------------------

        src_score = self.attn_src(
            h
        ).squeeze(-1).unsqueeze(2)

        dst_score = self.attn_dst(
            h
        ).squeeze(-1).unsqueeze(1)

        scores = self.leaky_relu(
            src_score + dst_score
        )

        # ----------------------------------------------------
        # Normalize attention across nodes
        # ----------------------------------------------------

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(
            attention
        )

        # ----------------------------------------------------
        # Weighted aggregation
        # ----------------------------------------------------

        output = torch.bmm(
            attention,
            h
        )

        # ----------------------------------------------------
        # Residual connection + normalization
        # ----------------------------------------------------

        output = self.norm(
            F.elu(output) + h
        )

        return output


# ============================================================
# GAT + LSTM
# ============================================================

class GAT_LSTM(nn.Module):

    def __init__(
        self,
        num_nodes=30,
        gat_hidden=16,
        lstm_hidden=64,
        dropout=0.1
    ):
        super().__init__()

        self.num_nodes = num_nodes

        self.gat_hidden = gat_hidden

        # ----------------------------------------------------
        # GAT
        # ----------------------------------------------------

        self.gat = GraphAttentionLayer(
            in_features=1,
            out_features=gat_hidden,
            dropout=dropout
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # GAT produces:
        #
        # num_nodes × gat_hidden
        #
        # For 30 nodes:
        #
        # 30 × 16 = 480
        #
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=num_nodes * gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Final prediction layer
        # ----------------------------------------------------

        self.output = nn.Linear(
            lstm_hidden,
            num_nodes
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Expected:
        #
        # [batch, time, nodes]
        # ----------------------------------------------------

        if x.ndim != 3:

            raise ValueError(
                "GAT_LSTM expects input "
                "[batch, time, nodes]. "
                f"Received {tuple(x.shape)}"
            )

        actual_nodes = x.size(-1)

        # ----------------------------------------------------
        # Explicit node-count validation
        # ----------------------------------------------------

        if actual_nodes != self.num_nodes:

            raise ValueError(
                f"GAT_LSTM configured for "
                f"{self.num_nodes} nodes, "
                f"but received {actual_nodes} nodes."
            )

        gat_sequence = []

        # ----------------------------------------------------
        # Apply GAT independently to every time step
        # ----------------------------------------------------

        for t in range(
            x.size(1)
        ):

            # [batch, nodes]
            xt = x[:, t, :]

            # [batch, nodes, 1]
            xt = xt.unsqueeze(-1)

            # [batch, nodes, gat_hidden]
            ht = self.gat(
                xt
            )

            # [batch, nodes * gat_hidden]
            ht = ht.reshape(
                x.size(0),
                -1
            )

            gat_sequence.append(
                ht
            )

        # ----------------------------------------------------
        # [batch, time, nodes * gat_hidden]
        # ----------------------------------------------------

        sequence = torch.stack(
            gat_sequence,
            dim=1
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        lstm_output, _ = self.lstm(
            sequence
        )

        # ----------------------------------------------------
        # Last time step
        # ----------------------------------------------------

        last = lstm_output[
            :, -1, :
        ]

        last = self.dropout(
            last
        )

        # ----------------------------------------------------
        # Predict one value per node
        # ----------------------------------------------------

        prediction = self.output(
            last
        )

        return prediction