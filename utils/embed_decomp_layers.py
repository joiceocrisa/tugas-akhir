import math
import numpy as np
import time
import torch
import torch.nn as nn

# Data Embedding ===================================================================================================================================
class TokenEmbedding(nn.Module):
    """
    Token Embedding untuk data memproyeksikan nilai input mentah (raw time series)
    ke dalam ruang representasi berdimensi lebih tinggi (`d_model`)
    menggunakan konvolusi 1D.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    """
    Fixed embedding pada FEDformer menggunakan representasi sinusoidal statis 
    untuk fitur waktu diskrit, sehingga informasi temporal dapat dimasukkan ke dalam model 
    tanpa menambah parameter yang perlu dilatih.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """
    Temporal embedding digunakan untuk merepresentasikan 
    informasi waktu diskrit seperti bulan, hari, dan jam ke dalam ruang vektor berdimensi `d_model`, 
    sehingga model dapat menangkap pola musiman dan periodik pada data deret waktu.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    """
    Time Feature Embedding untuk merepresentasikan fitur waktu kontinu
    (seperti jam, hari, bulan) ke dalam ruang vektor berdimensi `d_model`
    menggunakan proyeksi linier.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding_no_pos(nn.Module):
    """ 
    Data Embedding tanpa Positional Encoding.

    Modul ini menggabungkan 2 jenis embedding, yaitu:
    (1) Value embedding untuk memproyeksikan data numerik mentah
        ke dalam ruang representasi model,
    (2) Temporal embedding untuk mengodekan informasi waktu
        seperti jam, hari, dan bulan,

    Representasi akhir diperoleh dengan menjumlahkan embedding nilai dan embedding
    temporal, kemudian diterapkan regularisasi dropout untuk mengurangi risiko overfitting.

    Parameters
    ----------
    c_in : int
        Number of input channels (features) in the raw time series.
    d_model : int
        Dimension of the embedding space.
    embed_type : str, optional
        Type of temporal embedding ('fixed', 'learnable', or 'timeF').
        Default is 'fixed'.
    freq : str, optional
        Frequency of the time series data (e.g., 'h' for hourly, 'd' for daily).
        Default is 'h'.
    dropout : float, optional
        Dropout rate applied after embedding summation. Default is 0.1.

    Input
    -----
    x : torch.Tensor
        Input time series of shape [Batch, Sequence Length, c_in].
    x_mark : torch.Tensor
        Time feature tensor of shape [Batch, Sequence Length, d_time].

    Output
    ------
    torch.Tensor
        Embedded representation of shape [Batch, Sequence Length, d_model].
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_no_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)


# Series Decomposition ===================================================================================================================================
class moving_avg(nn.Module):
    """
    Blok Moving Average untuk ekstraksi tren pada data deret waktu (time series).
    Dilakukan menggunakan Average Pooling (AvgPool1d) dengan padding khusus pada kedua ujung sekuens,
    sehingga panjang output tetap sama dengan panjang input.

    Parameter
    ----------
    kernel_size : int
        Ukuran jendela moving average yang menentukan tingkat penghalusan tren.
        Nilai yang lebih besar menghasilkan tren yang lebih halus.
    stride : int
        Langkah pergeseran jendela pada operasi moving average.
        Umumnya diatur ke 1 untuk mempertahankan resolusi waktu.

    Input
    -----
    x : torch.Tensor
        Tensor input deret waktu dengan bentuk [Batch, Panjang Sekuens, Jumlah Fitur].

    Output
    ------
    torch.Tensor
        Tensor keluaran dengan bentuk [Batch, Panjang Sekuens, Jumlah Fitur],
        yang merepresentasikan komponen tren hasil operasi moving average.
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Dekomposisi deret waktu ini memisahkan deret waktu input menjadi dua komponen utama:
    (1) komponen tren (trend), dan
    (2) komponen residu/musiman (seasonal atau residual).

    Dekomposisi dilakukan menggunakan operasi moving average sebagai
    estimator tren. Komponen tren diperoleh dari hasil moving average,
    sedangkan komponen residu dihitung sebagai selisih antara sinyal
    asli dan tren.

    Blok ini merupakan komponen inti dalam arsitektur FEDformer, 
    yang memungkinkan pemodelan tren dan pola musiman dilakukan 
    secara terpisah untuk meningkatkan akurasi peramalan jangka panjang.

    Parameter
    ----------
    kernel_size : int
        Ukuran jendela moving average yang digunakan untuk mengekstraksi
        komponen tren dari deret waktu.

    Input
    -----
    x : torch.Tensor
        Tensor input deret waktu dengan bentuk
        [Batch, Panjang Sekuens, Jumlah Fitur].

    Output
    ------
    tuple of torch.Tensor
        - res : torch.Tensor
            Komponen residu atau musiman dengan bentuk
            [Batch, Panjang Sekuens, Jumlah Fitur].
        - moving_mean : torch.Tensor
            Komponen tren hasil moving average dengan bentuk
            [Batch, Panjang Sekuens, Jumlah Fitur].
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp_multi(nn.Module):
    """
    Blok dekomposisi deret waktu multi-skala 
    memperluas konsep series decomposition standar dengan
    menggunakan beberapa jendela moving average (multi-scale) secara
    simultan untuk mengekstraksi komponen tren deret waktu.
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 

def fedformer_decompose(
    series,
    kernel_sizes,
    DEVICE
):
    """
    series: numpy array [T] atau [T, 1]
    return: seasonal, trend (numpy)
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    x = torch.FloatTensor(series).unsqueeze(0).to(DEVICE)  # [1, T, 1]

    decomp = series_decomp_multi(kernel_sizes).to(DEVICE)
    decomp.eval()

    with torch.no_grad():
        seasonal, trend = decomp(x)

    return (
        seasonal.squeeze().cpu().numpy(),
        trend.squeeze().cpu().numpy()
    )


# Auto Correlation Layer ===================================================================================================================================
class AutoCorrelationLayer(nn.Module):
    """
    AutoCorrelation Layer sebagai pembungkus (wrapper) mekanisme korelasi/atensi
    pada FEDformer.

    Layer ini berfungsi sebagai antarmuka umum (interface) antara representasi
    embedding berdimensi `d_model` dengan mekanisme korelasi internal (FourierBlock, FourierCrossAttention).
    Layer ini bertugas melakukan:
    (1) Proyeksi linear Query, Key, dan Value,
    (2) Pemisahan ke dalam beberapa head (multi-head),
    (3) Pemanggilan mekanisme korelasi internal,
    (4) Penggabungan kembali output multi-head ke dimensi `d_model`.

    Parameter
    ----------
    correlation : nn.Module
        Modul korelasi internal yang akan digunakan, misalnya:
        - FourierBlock (self-correlation),
        - FourierCrossAttention (cross-correlation),
    d_model : int
        Dimensi embedding input dan output layer.
    n_heads : int
        Jumlah head pada mekanisme multi-head correlation.
    d_keys : int, optional
        Dimensi fitur untuk Query dan Key pada setiap head.
        Jika None, maka d_keys = d_model // n_heads.
    d_values : int, optional
        Dimensi fitur untuk Value pada setiap head.
        Jika None, maka d_values = d_model // n_heads.

    Input
    -----
    queries : torch.Tensor
        Tensor query dengan bentuk [Batch, Panjang Sekuens Query, d_model].
    keys : torch.Tensor
        Tensor key dengan bentuk [Batch, Panjang Sekuens Key, d_model].
    values : torch.Tensor
        Tensor value dengan bentuk [Batch, Panjang Sekuens Value, d_model].
    attn_mask : torch.Tensor or None
        Mask untuk mekanisme korelasi (opsional).

    Output
    ------
    tuple
        - torch.Tensor: Output layer dengan bentuk [Batch, Panjang Sekuens Query, d_model].
        - torch.Tensor or None: Informasi atensi/korelasi (jika disediakan oleh modul internal).
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# Layer Normalization ===================================================================================================================================
class my_Layernorm(nn.Module):
    """
    Layer Normalization khusus untuk komponen musiman (seasonal component).

    Tujuan menghilangkan bias global sepanjang dimensi waktu (time dimension).
    Setelah normalisasi fitur dilakukan, nilai rata-rata temporal dari
    output normalisasi dikurangkan kembali sehingga komponen musiman
    memiliki rata-rata nol pada setiap fitur.

    Pendekatan ini dirancang khusus untuk menjaga sifat musiman
    (zero-mean oscillation) dan mencegah kebocoran informasi tren
    ke dalam komponen musiman selama proses pembelajaran.

    Parameter
    ----------
    d_model : int
        Dimensi fitur yang akan dinormalisasi.

    Input
    -----
    x : torch.Tensor
        Tensor input dengan bentuk [Batch, Panjang Sekuens, Jumlah Fitur].

    Output
    ------
    torch.Tensor
        Tensor hasil normalisasi dengan rata-rata nol sepanjang
        dimensi waktu, berbentuk [Batch, Panjang Sekuens, Jumlah Fitur].
    """ 
    def __init__(self, d_model):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias