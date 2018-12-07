
import sys
path = '/var/www/drive'
if path not in sys.path:
    sys.path.append('/var/www/drive')

from drive import app as application
