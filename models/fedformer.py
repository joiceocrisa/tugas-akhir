"""

FEDformer Model
│
├── series_decomp
├── DataEmbedding (tanpa positional encoding)
│
├── Encoder
│   └── EncoderLayer
│       └── AutoCorrelationLayer
│           └── FourierBlock
│
├── Decoder
│   └── DecoderLayer
│       ├── AutoCorrelationLayer (FourierBlock)
│       └── AutoCorrelationLayer (FourierCrossAttention)
│
└── Projection

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fourier import Frequency_FourierBlock, Frequency_FourierCrossAttention
from utils.embed_decomp_layers import DataEmbedding_no_pos, series_decomp, series_decomp_multi, AutoCorrelationLayer, my_Layernorm

# Encoder ============================================================================================================================================
class EncoderLayer(nn.Module):
    """
    Encoder layer ini menggabungkan:
    (1) Mekanisme attention berbasis frekuensi (Fourier),
    (2) Feed-Forward Network berbasis konvolusi 1D,
    (3) Dua tahap dekomposisi deret waktu untuk memisahkan
        komponen musiman (seasonal) dan tren (trend).

    Encoder pada FEDformer secara eksplisit melakukan dekomposisi deret waktu setelah
    setiap operasi utama untuk menjaga pemisahan informasi
    musiman dan tren secara progresif.

    Parameter
    ----------
    attention : nn.Module
        Modul attention yang digunakan (misalnya FourierBlock
        yang dibungkus oleh AutoCorrelationLayer).
    d_model : int
        Dimensi representasi model (jumlah fitur internal).
    d_ff : int, opsional
        Dimensi hidden pada feed-forward network.
        Jika None, akan diset menjadi 4 * d_model.
    moving_avg : int atau list[int], opsional
        Ukuran kernel moving average untuk dekomposisi deret waktu.
        Jika berupa list, digunakan multi-scale decomposition.
    dropout : float, opsional
        Nilai dropout untuk regularisasi.
    activation : str, opsional
        Fungsi aktivasi yang digunakan ('relu' atau 'gelu').

    Input
    -----
    x : torch.Tensor
        Tensor input dengan bentuk
        [Batch, Panjang Sekuens, d_model].
    attn_mask : torch.Tensor, opsional
        Masking attention (biasanya None pada forecasting).

    Output
    ------
    res : torch.Tensor
        Representasi hasil encoder setelah dekomposisi,
        berbentuk [Batch, Panjang Sekuens, d_model].
    attn : torch.Tensor atau None
        Bobot attention (jika diaktifkan).
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class Encoder(nn.Module):
    """
    Encoder pada arsitektur FEDformer yang berfungsi untuk
    mengekstraksi representasi musiman (seasonal representation)
    dari deret waktu input melalui beberapa lapisan EncoderLayer.

    Encoder ini merupakan pengembangan dari encoder Transformer,
    di mana setiap EncoderLayer menggunakan mekanisme
    AutoCorrelation (misalnya Fourier-based correlation)
    dan progressive series decomposition untuk
    memisahkan komponen musiman dan tren.

    Encoder hanya mempertahankan komponen musiman sebagai
    representasi laten, sedangkan komponen tren secara eksplisit
    dibuang dan diproses di bagian decoder.

    Parameter
    ----------
    attn_layers : list[nn.Module]
        Daftar EncoderLayer yang akan disusun secara berurutan.
        Setiap EncoderLayer berisi self-attention (Fourier)
        dan feed-forward network dengan dekomposisi deret waktu.
    conv_layers : list[nn.Module], opsional
        Lapisan konvolusi tambahan antar EncoderLayer.
        Biasanya digunakan pada varian Autoformer,
        dan bernilai None pada FEDformer standar.
    norm_layer : nn.Module, opsional
        Lapisan normalisasi akhir (misalnya my_Layernorm)
        untuk menstabilkan representasi keluaran encoder.

    Input
    -----
    x : torch.Tensor
        Input encoder dengan bentuk [Batch, Panjang Sekuens, d_model].
    attn_mask : torch.Tensor, opsional
        Masking untuk self-attention (jarang digunakan pada time series).

    Output
    ------
    x : torch.Tensor
        Representasi musiman hasil encoder dengan bentuk
        [Batch, Panjang Sekuens, d_model].
    attns : list
        Daftar matriks attention (atau korelasi) dari setiap EncoderLayer.
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# Decoder  ============================================================================================================================================
class DecoderLayer(nn.Module):
    """
    Decoder layer ini berfungsi untuk:
    (1) Memproses informasi musiman (seasonal) dari decoder input
        menggunakan self-attention,
    (2) Menggabungkan informasi historis dari encoder
        melalui cross-attention,
    (3) Menghasilkan dan mengakumulasi komponen tren (trend)
        secara eksplisit melalui beberapa tahap dekomposisi.

    Berbeda dengan encoder yang hanya mempertahankan komponen
    musiman, decoder FEDformer secara eksplisit memodelkan
    dan mengakumulasi komponen tren sebagai bagian dari proses
    peramalan.

    Parameter
    ----------
    self_attention : nn.Module
        Modul self-attention decoder (misalnya FourierBlock
        yang dibungkus AutoCorrelationLayer).
    cross_attention : nn.Module
        Modul cross-attention antara decoder dan encoder.
    d_model : int
        Dimensi representasi internal model.
    c_out : int
        Jumlah variabel output (misalnya 1 untuk univariate).
    d_ff : int, opsional
        Dimensi hidden feed-forward network.
        Jika None, akan diset menjadi 4 * d_model.
    moving_avg : int atau list[int], opsional
        Ukuran kernel moving average untuk dekomposisi deret waktu.
        Jika berupa list, digunakan multi-scale decomposition.
    dropout : float, opsional
        Nilai dropout untuk regularisasi.
    activation : str, opsional
        Fungsi aktivasi non-linear ('relu' atau 'gelu').

    Input
    -----
    x : torch.Tensor
        Input decoder (komponen musiman) dengan bentuk
        [Batch, Panjang Sekuens Decoder, d_model].
    cross : torch.Tensor
        Output encoder sebagai memori historis,
        berbentuk [Batch, Panjang Sekuens Encoder, d_model].
    x_mask : torch.Tensor, opsional
        Masking self-attention decoder.
    cross_mask : torch.Tensor, opsional
        Masking cross-attention decoder–encoder.

    Output
    ------
    x : torch.Tensor
        Representasi musiman hasil decoder,
        berbentuk [Batch, Panjang Sekuens Decoder, d_model].
    residual_trend : torch.Tensor
        Komponen tren hasil akumulasi dari beberapa tahap
        dekomposisi, berbentuk [Batch, Panjang Sekuens Decoder, c_out].
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend

class Decoder(nn.Module):
    """
    Decoder pada arsitektur FEDformer yang bertugas untuk
    menghasilkan komponen musiman dan komponen tren secara eksplisit
    untuk keperluan peramalan deret waktu.

    Decoder FEDformer terdiri dari beberapa DecoderLayer yang disusun
    secara berurutan. Setiap DecoderLayer:
    (1) Memperbarui representasi musiman melalui self-attention
        dan cross-attention,
    (2) Mengekstraksi komponen tren residual melalui mekanisme
        progressive series decomposition,
    (3) Menambahkan tren residual tersebut ke tren global
        yang sedang dibangun.

    Dengan desain ini, FEDformer memisahkan proses pemodelan
    musiman (seasonal) dan tren (trend) secara eksplisit,
    sehingga peramalan akhir merupakan penjumlahan dari kedua komponen.

    Parameter
    ----------
    layers : list[nn.Module]
        Daftar DecoderLayer yang akan dieksekusi secara berurutan.
    norm_layer : nn.Module, opsional
        Lapisan normalisasi akhir untuk keluaran musiman decoder
        (misalnya my_Layernorm).
    projection : nn.Module, opsional
        Lapisan proyeksi akhir untuk mengubah dimensi keluaran
        musiman ke dimensi target output.

    Input
    -----
    x : torch.Tensor
        Input decoder (komponen musiman) dengan bentuk
        [Batch, Panjang Sekuens Decoder, d_model].
    cross : torch.Tensor
        Output encoder (memori historis) dengan bentuk
        [Batch, Panjang Sekuens Encoder, d_model].
    x_mask : torch.Tensor, opsional
        Mask untuk self-attention decoder.
    cross_mask : torch.Tensor, opsional
        Mask untuk cross-attention decoder–encoder.
    trend : torch.Tensor
        Komponen tren awal (hasil dekomposisi encoder),
        berbentuk [Batch, Panjang Sekuens Decoder, c_out].

    Output
    ------
    x : torch.Tensor
        Representasi musiman akhir decoder,
        berbentuk [Batch, Panjang Sekuens Decoder, c_out]
        (jika projection digunakan).
    trend : torch.Tensor
        Komponen tren akhir hasil akumulasi semua DecoderLayer,
        berbentuk [Batch, Panjang Sekuens Decoder, c_out].
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


# FEDformer Model ============================================================================================================================================

class FEDformer_Model(nn.Module):
    """
    Frequency Enhanced Decomposed Transformer (FEDformer) Model
    yang digunakan untuk melakukan peramalan deret waktu (time series),
    khususnya pada data dengan pola musiman dan tren yang kuat,
    seperti harga saham, beban listrik, atau data ekonomi.

    Model FEDformer mengombinasikan pendekatan dekomposisi deret waktu
    (trend dan seasonal) dengan mekanisme Transformer berbasis domain
    frekuensi (Fourier). Dengan memanfaatkan representasi frekuensi,
    FEDformer mampu menangkap ketergantungan jangka panjang secara
    lebih efisien dibandingkan Transformer konvensional di domain waktu.

    Parameter
    ----------
    configs : object
        Objek konfigurasi yang berisi seluruh hiperparameter model,
        antara lain:

        - enc_in : int  
          Jumlah fitur input pada encoder.

        - dec_in : int  
          Jumlah fitur input pada decoder.

        - c_out : int  
          Jumlah fitur keluaran (output), umumnya 1 untuk data univariat.

        - seq_len : int  
          Panjang sekuens input historis yang digunakan oleh encoder.

        - label_len : int  
          Panjang sekuens historis yang diberikan sebagai input awal
          pada decoder.

        - pred_len : int  
          Panjang horizon prediksi yang dihasilkan oleh model.

        - d_model : int  
          Dimensi representasi laten (embedding) pada Transformer.

        - d_ff : int  
          Dimensi feed-forward layer pada encoder dan decoder.

        - n_heads : int  
          Jumlah head pada mekanisme attention.

        - e_layers : int  
          Jumlah lapisan encoder.

        - d_layers : int  
          Jumlah lapisan decoder.

        - moving_avg : int atau list  
          Ukuran kernel moving average yang digunakan untuk
          dekomposisi tren dan musiman.
          Jika berupa list, maka digunakan multi-scale decomposition.

        - modes : int  
          Jumlah mode frekuensi Fourier yang dipilih.

        - mode_select : str  
          Metode pemilihan mode frekuensi
          (misalnya 'random' atau 'low').

        - dropout : float  
          Nilai dropout untuk regularisasi model.

        - activation : str  
          Fungsi aktivasi yang digunakan pada feed-forward network.

        - output_attention : bool  
          Menentukan apakah bobot attention dikembalikan
          sebagai bagian dari output model.

    Input
    -----
    x_enc : torch.Tensor
        Tensor input encoder berbentuk
        [Batch, seq_len, enc_in],
        yang merepresentasikan data historis utama.

    x_mark_enc : torch.Tensor
        Tensor penanda waktu (time features) untuk encoder
        dengan bentuk [Batch, seq_len, *],
        misalnya informasi hari, bulan, atau waktu.

    x_dec : torch.Tensor
        Tensor input decoder berbentuk
        [Batch, label_len + pred_len, dec_in],
        yang digunakan sebagai input awal decoder.

    x_mark_dec : torch.Tensor
        Tensor penanda waktu untuk decoder
        dengan bentuk [Batch, label_len + pred_len, *].

    enc_self_mask : torch.Tensor, opsional
        Mask untuk self-attention pada encoder.

    dec_self_mask : torch.Tensor, opsional
        Mask untuk self-attention pada decoder.

    dec_enc_mask : torch.Tensor, opsional
        Mask untuk cross-attention antara decoder dan encoder.

    Output
    ------
    out : torch.Tensor
        Tensor output prediksi berbentuk
        [Batch, pred_len, c_out],
        yang merepresentasikan hasil peramalan
        untuk horizon waktu ke depan.

    attns : list of torch.Tensor, opsional
        Bobot attention dari encoder,
        hanya dikembalikan jika `output_attention=True`.

    Catatan
    -------
    - Model melakukan dekomposisi deret waktu menjadi komponen
      tren (trend) dan musiman (seasonal) sebelum proses encoding.
    - Attention dihitung di domain frekuensi menggunakan
      transformasi Fourier untuk meningkatkan efisiensi
      pemodelan ketergantungan jangka panjang.
    - Output akhir diperoleh dengan menjumlahkan kembali
      komponen tren dan musiman hasil decoder.
    """
    def __init__(self, configs):
        super(FEDformer_Model, self).__init__()
        # inisialisasi parameter dasar model
        self.mode_select = configs.mode_select  # pemilihan frekuensi
        self.modes = configs.modes              # jumlah mode frekuensi yang dipilih
        self.seq_len = configs.seq_len          # panjang sekuens input
        self.label_len = configs.label_len      # panjang sekuens label (historis)
        self.pred_len = configs.pred_len        # panjang sekuens prediksi
        self.output_attention = configs.output_attention # apakah output attention diinginkan

        # Decomposition 
        kernel_size = configs.moving_avg # ukuran kernel moving average untuk dekomposisi
        if isinstance(kernel_size, list): 
            self.decomp = series_decomp_multi(kernel_size) # multi-scale decomposition
        else:
            self.decomp = series_decomp(kernel_size) # single-scale decomposition

        # Embedding
        self.enc_embedding = DataEmbedding_no_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed_type,
            freq=configs.freq,
            dropout=configs.dropout
        )   
        self.dec_embedding = DataEmbedding_no_pos(
            configs.dec_in,
            configs.d_model,
            configs.embed_type,
            freq=configs.freq,
            dropout=configs.dropout
        )

        # Attention 
        encoder_self_attention = Frequency_FourierBlock(
            d_model=configs.d_model,
            out_channels=configs.d_model,
            seq_len=self.seq_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select
        )
        decoder_self_attention = Frequency_FourierBlock(
            d_model=configs.d_model,
            out_channels=configs.d_model,
            seq_len=self.seq_len//2+self.pred_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select
        )
        decoder_cross_attention = Frequency_FourierCrossAttention(
            d_model=configs.d_model,
            out_channels=configs.d_model,
            seq_len_q=self.seq_len//2+self.pred_len,
            seq_len_kv=self.seq_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select
        )

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2 + configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_attention,
                        configs.d_model,
                        configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_attention,
                        configs.d_model,
                        configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_attention,
                        configs.d_model,
                        configs.n_heads),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out) 
        )
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # dekomposisi input
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zero = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]])
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0,0,0,self.pred_len)) 

        # encoder 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init
        )

        # final output
        dec_out = seasonal_part + trend_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :] 