"""
Render platform için özel konfigürasyon
"""
import os
from pathlib import Path

def get_render_config():
    """Render platform'da çalışırken gerekli ayarları yap"""
    
    # Render'da persistent storage yok, bu yüzden model dosyalarını memory'de tut
    if os.getenv('RENDER'):
        return {
            'model_storage': 'memory',
            'data_storage': 'memory',
            'log_level': 'INFO',
            'port': int(os.getenv('PORT', 8000))
        }
    
    # Local development
    return {
        'model_storage': 'disk',
        'data_storage': 'disk',
        'log_level': 'DEBUG',
        'port': 8000
    }
