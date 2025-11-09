from TMTrain4 import KappaLightning
from iwpc.utils import latest_ckpt
from bokeh.io import curdoc
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D
import torch 

module = KappaLightning.load_from_checkpoint(latest_ckpt("/Users/albaburgosmondejar/QMISID/dielectron_logs/kappa/version_41"))
curdoc().add_root(BokehFunctionVisualiser2D.visualise(module).root)
