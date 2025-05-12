class PlotLC:
    def __init__(self, data,err, time):
        from bokeh.layouts import column,row
        from bokeh.plotting import figure, show
        p1 = figure(width=1600, height=400, title="Lightcurve Sample",background_fill_color="#fafafa")
        p1.xgrid.grid_line_color = None
        for band,c in zip(range(data.shape[1]), ['red','green']):
            self.get_segments(data[:,band],err[:,band],time[:,band],p1, color = c)
            p1.scatter(x = time[:,band], y = data[:,band],size = 5, color = c)
        p1.xaxis.axis_label = 'Time MJD'
        p1.yaxis.axis_label = 'Flux'

        show(p1)
    @staticmethod
    def get_segments(data, err, time, p, color):
        for di, erri, ti, in zip(data,err, time):
            x0 = ti
            x1 = ti
            y0 =  di + erri
            y1 =  di -  erri
            p.segment(x0,y0,x1,y1, line_width = 1, color = color)
"""
import numpy as np
n = 200
time = np.linspace(0,100,n).reshape(n,1).repeat(2,axis = -1)
err = np.random.normal(0,8,size = (n,2))
data =np.random.randint(0,100, size = (n,2))


PlotLC(data,err, time)
"""