import plotly.graph_objects

oq_data = [
    {"ram": 11.40, "kld": 0.246338, "label": "oQ2"},
    {"ram": 14.96, "kld": 0.134155, "label": "oQ3"},
    {"ram": 16.17, "kld": 0.126221, "label": "oQ3.5"},
    {"ram": 18.98, "kld": 0.016815, "label": "oQ4"},
    {"ram": 22.76, "kld": 0.008270, "label": "oQ5"},
    {"ram": 26.51, "kld": 0.006718, "label": "oQ6"},
    {"ram": 34.27, "kld": 0.002277, "label": "oQ8"},
]

q_data = [
    {"ram": 10.10, "kld": 3.042969, "label": "Q2"},
    {"ram": 14.14, "kld": 0.206299, "label": "Q3"},
    {"ram": 18.17, "kld": 0.054230, "label": "Q4"},
    {"ram": 22.20, "kld": 0.015419, "label": "Q5", "pos": "top left"},
    {"ram": 26.23, "kld": 0.007050, "label": "Q6", "pos": "top left"},
    {"ram": 34.30, "kld": 0.000926, "label": "Q8", "pos": "top left", "x": -0.15},
    {"ram": 17.16, "kld": 0.097501, "label": "MXFP4"},
    {"ram": 33.29, "kld": 0.038293, "label": "MXFP8"},
]

x_min, x_max = 10, 36
y_min, y_max = 0, 0.3
height = 800
margin_top = 50
dx_minor = 1
dy_minor = 0.01
color_oq = "#1f77b4"
color_q = "#000000"
color_text = "#000000"
color_grid_major = "#d3d3d3"
color_grid_minor = "#f5f5f5"

def extract(data):
    ram = [d["ram"] for d in data]
    kld = [d["kld"] for d in data]
    labels = [d["label"] for d in data]
    textpos = [d.get("pos", "top right") for d in data]
    x_text = [d["ram"] + d.get("x", 0) for d in data]
    y_text = [d["kld"] + d.get("y", 0.0025) for d in data]
    return ram, kld, labels, textpos, x_text, y_text

ram_oq, kld_oq, labels_oq, pos_oq, x_oq_text, y_oq_text = extract(oq_data)
ram_q,  kld_q,  labels_q,  pos_q,  x_q_text,  y_q_text  = extract(q_data)

fig = plotly.graph_objects.Figure()

scatter_markers_oq = plotly.graph_objects.Scatter(
    x=ram_oq,
    y=kld_oq,
    mode="markers",
    name="oQ",
    marker={"color": color_oq, "size": 8},
)
fig.add_trace(scatter_markers_oq)

scatter_markers_q = plotly.graph_objects.Scatter(
    x=ram_q,
    y=kld_q,
    mode="markers",
    name="Q",
    marker={"color": color_q, "size": 8},
)
fig.add_trace(scatter_markers_q)

scatter_labels_oq = plotly.graph_objects.Scatter(
    x=x_oq_text,
    y=y_oq_text,
    mode="text",
    name="oQ_labels",
    text=labels_oq,
    textposition=pos_oq,
    textfont={"color": color_oq, "size": 14},
    showlegend=False,
)
fig.add_trace(scatter_labels_oq)

scatter_labels_q = plotly.graph_objects.Scatter(
    x=x_q_text,
    y=y_q_text,
    mode="text",
    name="Q_labels",
    text=labels_q,
    textposition=pos_q,
    textfont={"color": color_q, "size": 14},
    showlegend=False,
)
fig.add_trace(scatter_labels_q)

x_range = x_max - x_min
y_range = y_max - y_min
width = (height - margin_top) * (x_range / dx_minor) / (y_range / dy_minor)

fig.update_layout(
    title="Qwen3.6-35B-A3B",
    title_x=0.5,
    width=width,
    height=height,
    showlegend=False,
    font={"color": color_text},
    xaxis_title="RAM (GiB)",
    yaxis_title="KL Divergence (nats)",
    yaxis={
        "range": [y_min, y_max],
        "showgrid": True,
        "gridcolor": color_grid_major,
        "ticklabelposition": "outside",
        "ticklabelstandoff": 10,
        "minor": {
            "showgrid": True,
            "dtick": dy_minor,
            "gridcolor": color_grid_minor,
            "gridwidth": 0.5,
        },
    },
    xaxis={
        "range": [x_min, x_max],
        "showgrid": True,
        "gridcolor": color_grid_major,
        "ticklabelposition": "outside",
        "ticklabelstandoff": 10,
        "minor": {
            "showgrid": True,
            "dtick": dx_minor,
            "gridcolor": color_grid_minor,
            "gridwidth": 0.5,
        },
    },
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin={
        "t": margin_top,
        "b": 0,
        "l": 0,
        "r": 0,
    },
)

# fig.show()
fig.write_image("results/Qwen3.6-35B-A3B.svg")
