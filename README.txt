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

CHANGELOG

na8st-1.1 (1.1)

  * Pointing out discrepancies in \bar{g}_{Na} computations.
    Thanks to Steffen Platschek for pointing this out.

 -- Christoph Schmidt-Hieber <c.schmidt-hieber@ucl.ac.uk>  Sun, 20 Oct 2011 13:41:23 +0000

20150903 a patch (contributed by Arndt Roth and emailed by Christoph
Schmidt-Hieber) that adds temperature dependence was applied to
mod/na8st.mod.
