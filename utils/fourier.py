import numpy as np
import torch
import torch.nn as nn

def frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Menentukan indeks frekuensi yang digunakan pada domain Fourier
    untuk mekanisme attention berbasis frekuensi (Fourier Attention).

    Fungsi ini memilih sejumlah mode frekuensi dari hasil transformasi Fourier
    (FFT) berdasarkan panjang sekuens input. Mode frekuensi yang dipilih
    dapat diambil secara acak atau dengan memilih frekuensi rendah (low-frequency),
    tergantung metode seleksi yang digunakan.

    Parameter
    ---------
    seq_len : int
        Panjang sekuens waktu (sequence length) dari data input.
    modes : int, optional
        Jumlah mode frekuensi yang ingin dipilih.
        Nilai maksimum dibatasi oleh seq_len // 2.
        Default adalah 64.
    mode_select_method : str, optional
        Metode pemilihan mode frekuensi:
        - 'random' : memilih mode frekuensi secara acak,
        - selain itu : memilih mode frekuensi terendah (low-frequency).
        Default adalah 'random'.

    Returns
    -------
    list of int
        Daftar indeks mode frekuensi terpilih yang telah diurutkan,
        yang akan digunakan dalam operasi Fourier Attention.
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class Frequency_FourierBlock(nn.Module):
    """
    Frequency Fourier untuk mengimplementasikan mekanisme representasi berbasis domain frekuensi
    menggunakan Fast Fourier Transform (FFT). Tidak melakukan attention 
    di domain waktu, blok ini memproyeksikan sinyal ke domain frekuensi,
    melakukan transformasi linear kompleks pada sejumlah mode frekuensi terpilih,
    kemudian mengembalikannya ke domain waktu melalui inverse FFT (iFFT).

    Frequency_FourierBlock dirancang untuk menangkap pola global dan periodik
    pada data deret waktu.

    Parameter
    ----------
    d_model : int
        Dimensi fitur input (embedding dimension), biasanya sama dengan d_model
        pada Encoder/Decoder FEDformer.
    out_channels : int
        Dimensi fitur output setelah transformasi Fourier.
        Umumnya disamakan dengan d_model.
    seq_len : int
        Panjang sekuens waktu input.
        Digunakan untuk menentukan resolusi frekuensi FFT.
    modes : int, optional
        Jumlah mode frekuensi yang dipilih untuk diproses.
        Jika 0, jumlah mode ditentukan oleh fungsi `frequency_modes`.
        Default adalah 0.
    mode_select_method : str, optional
        Metode pemilihan mode frekuensi:
        - 'random' : memilih mode frekuensi secara acak,
        - selain itu : memilih frekuensi rendah (low-frequency).
        Default adalah 'random'.

    Input
    -----
    q : torch.Tensor
        Query tensor dengan bentuk [Batch, Panjang Sekuens, Jumlah Head, Dimensi Head].
    k : torch.Tensor
        Key tensor (tidak digunakan secara eksplisit pada blok ini).
    v : torch.Tensor
        Value tensor (tidak digunakan secara eksplisit pada blok ini).
    mask : torch.Tensor or None
        Mask perhatian (tidak digunakan dalam Fourier block).

    Output
    ------
    tuple
        - torch.Tensor: Output tensor hasil inverse FFT dengan bentuk
          [Batch, Jumlah Head, Dimensi Head, Panjang Sekuens].
        - None: Placeholder untuk attention weights (tidak digunakan).
    """

    def __init__(self, d_model, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(Frequency_FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (d_model * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, d_model // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1) 
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain 
        x = torch.fft.irfft(out_ft, n=x.size(-1)) 
        return (x, None) 

class Frequency_FourierCrossAttention(nn.Module):
    """
    Frequency Fourier Cross Attention mengimplementasikan mekanisme cross-attention pada domain frekuensi
    menggunakan Transformasi Fourier Cepat (FFT). Berbeda dengan self-attention
    biasanya yang bekerja di domain waktu, blok ini memproyeksikan query,
    key, dan value ke domain frekuensi, melakukan interaksi antar frekuensi
    melalui operasi atensi, kemudian mengembalikan hasilnya ke domain waktu
    menggunakan inverse FFT (iFFT).

    Frequency_FourierCrossAttention digunakan pada decoder FEDformer untuk
    memodelkan hubungan antara representasi decoder (query) dan output encoder
    (key-value) secara efisien dengan kompleksitas O(N).

    Parameter
    ----------
    d_model : int
        Dimensi embedding input (jumlah fitur total sebelum dibagi ke head).
    out_channel : int
        Dimensi embedding output, umumnya sama dengan d_model.
    seq_len_q : int
        Panjang sekuens query (decoder input).
    seq_len_kv : int
        Panjang sekuens key dan value (encoder output).
    modes : int, optional
        Jumlah mode frekuensi yang dipilih pada domain Fourier.
        Default adalah 64.
    mode_select_method : str, optional
        Metode pemilihan mode frekuensi:
        - 'random' : memilih frekuensi secara acak,
        - selain itu : memilih frekuensi rendah.
        Default adalah 'random'.
    activation : str, optional
        Fungsi aktivasi pada domain frekuensi:
        - 'tanh' : aktivasi hiperbolik,
        - 'softmax' : normalisasi probabilistik antar frekuensi.
        Default adalah 'tanh'.
    policy : int, optional
        Parameter tambahan (tidak digunakan secara eksplisit).
        Disediakan untuk kompatibilitas dengan eksperimen lanjutan.

    Input
    -----
    q : torch.Tensor
        Query tensor dengan bentuk [Batch, Panjang Sekuens Query, Jumlah Head, Dimensi Head].
    k : torch.Tensor
        Key tensor dengan bentuk [Batch, Panjang Sekuens Key, Jumlah Head, Dimensi Head].
    v : torch.Tensor
        Value tensor dengan bentuk [Batch, Panjang Sekuens Value, Jumlah Head, Dimensi Head].
    mask : torch.Tensor or None
        Mask atensi (tidak digunakan dalam implementasi ini).

    Output
    ------
    tuple
        - torch.Tensor: Output cross-attention pada domain waktu dengan bentuk
          [Batch, Jumlah Head, Dimensi Head, Panjang Sekuens Query].
        - None: Placeholder untuk attention weights.
    """

    def __init__(self, d_model, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(Frequency_FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.d_model = d_model
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (d_model * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, d_model // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} activation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.d_model / self.out_channels, n=xq.size(-1))
        return (out, None)
    






