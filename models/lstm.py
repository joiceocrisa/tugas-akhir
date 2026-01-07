import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Model Long Short-Term Memory (LSTM) yang digunakan untuk
    melakukan peramalan deret waktu (time series), khususnya
    pada prediksi harga saham.

    Model ini bekerja dengan memproses sekuens data historis
    untuk menangkap ketergantungan jangka pendek dan jangka panjang
    melalui mekanisme gate pada LSTM, kemudian menghasilkan
    nilai prediksi berdasarkan representasi tersembunyi
    pada langkah waktu terakhir.

    Parameter
    ----------
    input_size : int
        Jumlah fitur input pada setiap langkah waktu.
        Untuk data univariat, nilainya biasanya 1.

    hidden_size : int
        Jumlah unit tersembunyi (hidden units) pada lapisan LSTM.
        Parameter ini menentukan kapasitas model dalam
        mempelajari pola temporal data.

    num_layers : int
        Jumlah lapisan LSTM yang ditumpuk (stacked).
        Semakin banyak lapisan, semakin kompleks representasi
        temporal yang dapat dipelajari oleh model.

    output_size : int
        Jumlah nilai keluaran yang diprediksi oleh model.
        Umumnya bernilai 1 untuk prediksi satu langkah ke depan.

    dropout : float, opsional
        Nilai dropout yang diterapkan antar lapisan LSTM
        untuk mengurangi risiko overfitting.
        Dropout hanya aktif jika jumlah lapisan LSTM > 1.

    Input
    -----
    x : torch.Tensor
        Tensor input berbentuk
        [Batch, Panjang Sekuens, input_size],
        yang merepresentasikan data deret waktu historis.

    Output
    ------
    out : torch.Tensor
        Tensor output berbentuk
        [Batch, output_size],
        yang merepresentasikan hasil prediksi
        pada langkah waktu selanjutnya.

    Catatan
    -------
    - Hidden state dan cell state diinisialisasi dengan nol
      pada setiap proses forward.
    - Hanya output LSTM pada langkah waktu terakhir
      yang digunakan untuk menghasilkan prediksi.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Melakukan forward pass pada model LSTM.

        Parameter
        ---------
        x : torch.Tensor
            Tensor input dengan bentuk:
            (batch_size, sequence_length, input_size)

            di mana:
            - batch_size adalah jumlah sampel dalam satu batch,
            - sequence_length adalah panjang jendela deret waktu,
            - input_size adalah jumlah fitur pada setiap langkah waktu.

        Returns
        -------
        torch.Tensor
            Tensor output dengan bentuk:
            (batch_size, output_size)

            Tensor ini merepresentasikan nilai prediksi
            untuk setiap sekuens input.
        
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

