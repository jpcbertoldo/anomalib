

def test_aupimo_plots(anomaly_maps, masks, ubound):
    aupimo = AUPImO(num_threshs=1000, ubound=ubound)
    aupimo.update(anomaly_maps, masks)
    aupimo.compute()

    fig, ax = aupimo.plot()
    assert fig is not None
    assert ax is not None
    aupimo.plot(ax=ax)

    fig, ax = aupimo.plot_all_pimo_curves()
    assert fig is not None
    assert ax is not None
    aupimo.plot_all_pimo_curves(ax=ax)

    fig, ax = aupimo.plot_boxplot()
    assert fig is not None
    assert ax is not None
    aupimo.plot_boxplot(ax=ax)

    fig, ax = aupimo.plot_boxplot_pimo_curves()
    assert fig is not None
    assert ax is not None
    aupimo.plot_boxplot_pimo_curves(ax=ax)

    fig, ax = aupimo.plot_perimg_fprs()
    assert fig is not None
    assert ax is not None
    aupimo.plot_perimg_fprs(ax=ax)
