def l4():
    import plotly
    import plotly.graph_objs as go
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    N = 8

    # графики
    def get_plotting(data: list, title: str):
        data = np.array(data)
        length = data.shape[0]
        width = data.shape[1]
        x, y = np.meshgrid(np.arange(length), np.arange(width))
        fig = plt.figure(figsize=(10, 6), dpi=80)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x, y, data, color="b")
        ax.view_init(20, -120)
        plt.title(title)
        plt.show()

    # преобразование Адамара
    def adamar(rows: list, is_forward: bool):
        if is_forward:
            new_rows = [rows[0], rows[4], rows[2], rows[6], rows[1], rows[5], rows[3], rows[7]]
        else:
            new_rows = rows.copy()

        delta = 4

        for i in range(0, delta):
            rows[i] = new_rows[i] + new_rows[i + delta]

        for i in range(delta, delta * 2):
            rows[i] = -new_rows[i] + new_rows[i - delta]

        new_rows = rows.copy()
        delta = 2

        for i in range(0, 2):
            rows[i] = new_rows[i] + new_rows[i + delta]

        for i in range(2, 4):
            rows[i] = -new_rows[i] + new_rows[i - delta]

        for i in range(4, 6):
            rows[i] = new_rows[i] - new_rows[i + delta]

        for i in range(6, 8):
            rows[i] = new_rows[i] + new_rows[i - delta]

        new_rows = rows.copy()

        rows[0] = new_rows[0] + new_rows[1]
        rows[1] = new_rows[0] - new_rows[1]
        rows[2] = new_rows[2] - new_rows[3]
        rows[3] = new_rows[2] + new_rows[3]
        rows[4] = new_rows[4] + new_rows[5]
        rows[5] = new_rows[4] - new_rows[5]
        rows[6] = new_rows[6] - new_rows[7]
        rows[7] = new_rows[6] + new_rows[7]

        if is_forward:
            return rows
        else:
            return np.array([rows[0], rows[7], rows[4], rows[3],
                             rows[2], rows[5], rows[6], rows[1]], dtype=np.float64)

    def adamar_transorm(current_matrix: np.array, is_forward: bool) -> np.array:
        current_matrix = np.array(current_matrix)

        matrix = current_matrix.copy()

        for i in range(0, matrix.shape[0]):
            matrix[i] = adamar(matrix[i], is_forward)

        for i in range(0, matrix.shape[1]):
            matrix[:, i] = adamar(matrix[:, i], is_forward)

        if not is_forward:
            matrix[:, :] = matrix[:, :] / (matrix.shape[0] ** 2)

        return matrix

    matrix_1 = [[2, 3, 2, 0, 5, 0, 0, 0],
                [0, 3, 3, 2, 1, 2, 0, 0],
                [0, -1, 2, 4, 2, -1, 0, 0],
                [0, 3, 1, 3, -1, 1, 4, 0],
                [0, 0, 6, 5, -1, 3, 0, 0],
                [0, 4, 0, 5, 0, 3, 0, 0],
                [0, 0, 4, 0, 4, 4, 0, 0],
                [0, 3, 4, 5, 6, 7, 5, 6]]

    matrix_2 = [[2, 3, 2, 0, 5, 0, 0, 0],
                [0, 3, 3, 2, 1, 2, 0, 0],
                [0, -1, 2, 4, 2, -1, 0, 0],
                [0, 3, 1, 3, -1, 1, 1, 0],
                [0, 0, 6, 5, -1, 3, 0, 0],
                [0, 4, 9, 5, 0, 3, 0, 0],
                [0, 0, 4, 0, 4, 4, 0, 0],
                [0, 3, 4, 5, 6, 7, 5, 6]]

    matrix_3 = [[2, 3, 2, 0, 5, 0, 0, 0],
                [0, 3, 3, 9, 1, 2, 0, 0],
                [0, -1, 2, 4, 2, -1, 0, 0],
                [0, 3, 1, 3, -1, 1, 4, 0],
                [0, 0, 6, 5, -1, 3, 0, 0],
                [0, 4, 0, 5, 0, 3, 0, 0],
                [0, 0, 4, 0, 4, 4, 0, 0],
                [0, 3, 4, 5, 6, 7, 5, 6]]

    get_plotting(matrix_1, 'Сигнал без помехи')

    get_plotting(matrix_2, 'Сигнал с первой единичной помехой')

    get_plotting(matrix_3, 'Сигнал со второй единичной помехой')

    new_matrix_1 = adamar_transorm(matrix_1, True)
    spectrum_matrix_1 = np.multiply(new_matrix_1, N ** (-2))
    get_plotting(spectrum_matrix_1, 'Спектр двумерного преобразования Адамара первого сигнала')
    print(np.array2string(spectrum_matrix_1, precision=3))

    new_matrix_2 = adamar_transorm(matrix_2, True)
    spectrum_matrix_2 = np.multiply(new_matrix_2, N ** (-2))
    get_plotting(spectrum_matrix_2, 'Спектр двумерного преобразования Адамара второго сигнала')
    print(np.array2string(spectrum_matrix_2, precision=3))

    new_matrix_3 = adamar_transorm(matrix_3, True)
    spectrum_matrix_3 = np.multiply(new_matrix_3, N ** (-2))
    get_plotting(spectrum_matrix_3, 'Спектр двумерного преобразования Адамара третьего сигнала')
    print(np.array2string(spectrum_matrix_3, precision=3))

    filter_matrix = np.linalg.solve(spectrum_matrix_2, spectrum_matrix_1)
    get_plotting(filter_matrix, 'Фильтр, который при умножении спектра 2 сигнала приводит его к спектру 1 сигнала')
    print(np.array2string(filter_matrix, precision=3))

    new_spectrum_matrix_2 = np.matmul(spectrum_matrix_2, filter_matrix)
    new_spectrum_matrix_2 = np.multiply(new_spectrum_matrix_2, N ** (2))

    get_plotting(new_spectrum_matrix_2, 'Умноженный второй сигнал на полученный фильтр')
    print(np.array2string(new_spectrum_matrix_2, precision=3))

    new_spectrum_matrix_3 = np.matmul(spectrum_matrix_3, filter_matrix)
    new_spectrum_matrix_3 = np.multiply(new_spectrum_matrix_3, N ** (2))
    get_plotting(new_spectrum_matrix_3, 'Умноженный третий сигнал на полученный фильтр')
    print(np.array2string(new_spectrum_matrix_3, precision=3))

    result_matrix_2 = adamar_transorm(new_spectrum_matrix_2, False)
    result_matrix_2 = np.around(result_matrix_2, decimals=3)
    get_plotting(result_matrix_2, 'Обратное преобразование Адамара для 2 сигнала')
    print(np.array2string(result_matrix_2, precision=3))

    result_matrix_3 = adamar_transorm(new_spectrum_matrix_3, False)
    result_matrix_3 = np.around(result_matrix_3, decimals=3)
    get_plotting(result_matrix_3, 'Обратное преобразование Адамара для 3 сигнала')
    print(np.array2string(result_matrix_3, precision=3))


if __name__ == '__main__':
    l4()