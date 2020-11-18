import numpy as np
import matplotlib.pyplot as plt


category_names = ['0', '1']
ones_zeros = {"ad": [2832, 15581], "ae": [8964, 9772], "ar": [9000, 9737], "at": [8531, 10069], "au": [7426, 10819], "be": [5193, 11492], "bg": [9149, 10085], "bo": [8308, 8611], "br": [6429, 9635], "ca": [7373, 10812], "ch": [9793, 11671], "cl": [6959, 9989], "co": [7934, 8644], "cr": [5885, 9326], "cy": [9364, 9662], "cz": [8358, 9529], "de": [8142, 11185], "dk": [7578, 8541], "do": [5840, 7975], "dz": [9401, 9708], "ec": [7007, 8271], "ee": [8595, 9525], "eg": [9536, 9980], "es": [8726, 12482], "fi": [7256, 11809], "fr": [7489, 10859], "gb": [7875, 10876], "gr": [4892, 12890], "gt": [8073, 10377], "hk": [5968, 12741], "hn": [9032, 10094], "hu": [6345, 9974], "id": [6804, 8213], "ie": [9168, 11474], "il": [
    5097, 11055], "in": [6659, 11269], "is": [5954, 11048], "it": [7685, 10362], "jp": [3003, 12260], "kw": [7865, 8221], "kz": [3256, 13835], "lt": [7715, 9323], "lu": [7529, 10782], "lv": [7921, 9447], "mc": [2574, 14993], "mt": [8845, 9848], "mx": [6303, 11341], "my": [7950, 9870], "ni": [7494, 8640], "nl": [5120, 9948], "no": [6712, 10724], "nz": [7966, 10438], "pa": [7731, 8759], "pe": [7209, 9285], "ph": [8027, 8632], "pl": [8134, 8490], "pt": [7027, 10037], "py": [8823, 9921], "ru": [6547, 12755], "sa": [7890, 8585], "se": [6554, 11764], "sg": [8086, 10430], "sk": [8166, 8954], "sv": [6378, 9013], "tr": [5471, 13594], "tw": [3166, 13719], "us": [4235, 14518], "uy": [6099, 8189], "za": [7408, 9540]}

results = {}
for k, v in ones_zeros.items():
    s = v[0]+v[1]
    zeros = (v[0]/s)*0.01
    ones = (v[1]/s)*0.01
    results[k] = [zeros, ones]


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()
