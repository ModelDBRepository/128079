This code accompanies the publication:

Schmidt-Hieber C, Bischofberger J. (2010)
Fast sodium channel gating supports localized and efficient 
axonal action potential initiation.
J Neurosci 30:10233-42

Send comments to c.schmidt-hieber_at_ucl.ac.uk

INSTALLATION

1. Install the required Python modules:
NumPy (http://numpy.scipy.org)
SciPy (http://www.scipy.org/SciPy)
Matplotlib (http://matplotlib.sourceforge.net/index.html)

Installation instructions can be found on the respective web sites.

2. NEURON and Python
Install NEURON with Python support. At the time of writing,
the binary distribution won't do and you'll have to compile
NEURON from source.

See 
http://www.davison.webfactional.com/notes/installation-neuron-python/
for instructions.

3. Optional
To solve kinetic gating schemes without NEURON, you'll need pyqmatrix:

http://code.google.com/p/pyqmatrix

USAGE

1. Compile the mechanism files in the mod directory:
$ nrnivmodl mod

2. Run the example simulations:

# Simple current-clamp simulation
$ python ap.py

# Voltage-clamp simulation comparing Q-Matrix and NEURON results:
$ python qmatrix.py

