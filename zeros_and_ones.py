import numpy as np
import matplotlib.pyplot as plt
from iso_countrynames_map import isocountry


# print(isocountry["AE"])

category_names = ['not popular', 'popular']
ones_zeros = {
    isocountry["gb".upper()]: [7875, 10876], isocountry["fr".upper()]: [7489, 10859], isocountry["it".upper()]: [7685, 10362], isocountry["ch".upper()]: [9793, 11671],
    isocountry["ae".upper()]: [8964, 9772], isocountry["at".upper()]: [8531, 10069], isocountry["bg".upper()]: [9149, 10085], isocountry["co".upper()]: [7934, 8644], isocountry["dz".upper()]: [9401, 9708], isocountry["ee".upper()]: [8595, 9525], isocountry["eg".upper()]: [9536, 9980], isocountry["hn".upper()]: [9032, 10094], isocountry["kw".upper()]: [7865, 8221], isocountry["mt".upper()]: [8845, 9848], isocountry["ni".upper()]: [7494, 8640], isocountry["sa".upper()]: [7890, 8585]
}

results = {}
for k, v in ones_zeros.items():
    s = v[0]+v[1]
    zeros = (v[0]/s)*100
    ones = (v[1]/s)*100
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
