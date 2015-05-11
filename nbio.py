# utility for displaying things in Ipython

from IPython.display import Image as show_img
from IPython.display import display_html, HTML, display
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import resample
from scipy.interpolate  import interp1d
from base64 import b64encode
import urllib
import types
import sympy
from itertools import cycle
import base64, struct
import sys
print 'loaded nbio'


DPI =80.

def show(I, *args, **kwargs):
    options = kwargs.get('options') or []
    s = Converter(options=options, default_options=kwargs)(I)
    #print s
    display_html(HTML(s))


def play(I, rate=44100):
    """
    create an embedded sound player in NB
    """
    rate0 = 44100
    if rate!=rate0:
        newshape = int(I.shape[0]*rate0/rate)
        I = fast_resample(I, newshape) #resample
    if str(I.dtype).startswith('float'):
        mx = I.max()
        I = (I/mx * 8800.).astype('int16')
        print I.max(), I.min()

    #c = Converter()
    #a = c.html_audio(I)
    wavPlayer(I, rate0)
    #display_html(HTML(a))

def wavPlayer(data, rate, scale=False, autoplay=False):
    """This method will display html 5 player for compatible browser with 
    embedded base64-encoded WAV audio data.

    Parameters :
    ------------
    data : 1d np.ndarray containing the audio data to be played
    rate : the data rate in Hz
    scale : if set to True, the audio signal is amplified to cover the full scale.
    """
    #if np.max(abs(data)) > 1 or scale:
    #    data = data/np.max(abs(data))
    #data = (2**13*data).astype(np.int16)
    
    buffer = BytesIO()
    buffer.write(b'RIFF')
    buffer.write(b'\x00\x00\x00\x00')
    buffer.write(b'WAVE')
    
    buffer.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    
    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    buffer.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<i', data.nbytes))

    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    buffer.write(data.astype(np.int16).tostring())

    # Determine file size and place it in correct position at start of the file.
    size = buffer.tell()
    buffer.seek(4)
    buffer.write(struct.pack('<i', size-8))
    
    val = buffer.getvalue()
    autoplay = " autoplay=\"autoplay\""*autoplay + ""
    
    src = """<audio controls="controls" style="width:600px"{autoplay}>
      <source controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>""".format(base64=base64.b64encode(val).decode("ascii"), autoplay=autoplay)
    display(HTML(src))



def to_cell(cell, **kwargs):
    """ """

    sa = kwargs.get('css') or {}

    style_str =  ''.join('%s:%s;'%(k,v) for k,v in sa.iteritems())
    # ---
    ta = tag_attribs = {}
    if style_str: ta['style'] = style_str
    attribs_str = ' '.join('%s="%s"'%(k, v) for k,v in ta.iteritems())
    # ---
    if kwargs.get('header_row'): tag_type = 'TH'
    else: tag_type = 'TD'
    tag = '<%s %s>'%(tag_type, attribs_str,)
    tag+= str(cell)
    tag+= '</%s>'%(tag_type,)

    return tag



def to_row(row, **kwargs):
    """ """
    sa = kwargs.get('css') or {}
    child_css = sa.get('TD') or [{}]

    style_str = ''.join('%s:%s;'%(k,v) for k,v in sa.iteritems() 
        if k not in ['TD'])

    # ---
    ta = tag_attribs = {}
    if style_str: ta['style'] = style_str
    attribs_str = ' '.join('%s="%s"'%(k, v) for k,v in ta.iteritems())
    # ---     

    css_iter = cycle(child_css)
    tag = '<TR %s>'%(attribs_str,)
    for cell in row:
        css = css_iter.next()

        tag+= to_cell(cell, css=css)
    tag+= '</TR>'
    return tag

def to_table(rows, **kwargs):
    """ """
    # -- css --

    sa = kwargs.get('css') or {}
    child_css = sa.get('TR') or [{}]


    style_str =  ''.join('%s:%s;'%(k,v) for k,v in sa.iteritems() 
        if k not in ['TR'])
    # -- attribs --
    ta = tag_attribs = {}
    if style_str: ta['style'] = style_str
    attribs_str = ' '.join('%s="%s"'%(k, v) for k,v in ta.iteritems())
    # -- child options --

    css_iter = cycle(child_css)
    # --- tag ---
    tag = '<TABLE %s>'%(attribs_str,)    
    for row in rows:
        css = css_iter.next()
        tag+= to_row(row, css=css,)
    tag+= '</TABLE>'
    return tag





def cycle2(v):
    try:
        if type(v) is str: raise TypeError
        return cycle(v)
    except TypeError:
        return cycle([v])



t_opts = dict(border='1', style=None, width=None,
    cellspacing=None, cellpadding=0, attribs=None, header_row=None,
    col_width=None, col_align='center', col_valign=None,
    col_char=None, col_charoff=None, col_styles=None, bgcolor=None,
    rulesthick=1, rulescolor='black', bordercolor='black', css={})


class Converter:
    def __init__(self, options=[], default_options={}):
        self.pool = []
        self.all_options = dict((id(e[0]),e[1]) for e in options)
        self.all_options['default'] = default_options
        self.style = '''<style type="text/css" scoped>
td {padding:0; border:1px solid black;}
th {padding:0; }
table { }
code {background-color:green; }
</style>'''
        #self.outer = True

    def __call__(self, I):
        s = self.html_all(I)
        return s


    def html_all(self, I, opts=None):
        """
        """
        i = id(I)
        if i in self.pool: 
            I = '...'

        self.pool.append(i)
        
        if opts is not None:
            pass
        else:
            if i in self.all_options:
                opts = self.all_options['default'].copy()
                opts.update(self.all_options[i])
            else:
                opts = self.all_options['default']
            #opts['outer'] = outer

        if None:
            pass
        elif isinstance(I, np.ndarray):
            s = self.html_array(I, opts)
        elif isinstance(I, Image.Image): 
            s = self.html_pil(I, opts)
        elif isinstance(I, plt.Figure):
            s = self.html_plot(I, opts)
        elif isinstance(I, sympy.Function) or isinstance(I, sympy.Equality) or isinstance(I,sympy.Expr):
            s = self.html_expr(I, opts)
        elif isinstance(I, dict) or type(I) is types.DictProxyType:
            s = self.html_dict(I, opts)
        elif isinstance(I, list):
            
            s = self.html_list(I, opts)
        elif hasattr(I, '__dict__'):
            s = self.html_object(I, opts)

        else:
            s = '<code>%s</code>'%(str(I),)
        self.pool.pop()
        return s
  
    def html_expr(self, sympy_expression, opts):
        """ """
        preamble = "\\documentclass[12pt]{article}\n"\
                    "\\pagestyle{empty}\n"\
                    "\\usepackage[margin=0.0in]{geometry}\n"\
                    "\\usepackage{amsmath,amsfonts}\n"\
                    "\\begin{document}"
        bio = BytesIO()
        sympy.printing.preview(sympy_expression, 
            output='png', 
            viewer='BytesIO', outputbuffer=bio,
            dvioptions = ['-D', '160', '-Q 9', '-bg', 'Transparent'],
            preamble = preamble,
        )
        bio.seek(0)
        return self.html_image(bio)


    def html_object(self, obj, opts):
        """ """
        return self.html_all(obj.__dict__, opts)

    def html_dict(self, d, opts):
        l = [[k,v] for (k,v) in d.iteritems()]
        if opts.get('orient')=='row':
            l = zip(*l)
        return self.html_all(l, opts)

    def html_list(self, l, opts):

        excepts = [str, np.ndarray]
        allowed = [list, tuple]
        try:
            for e in l:
                if type(e) is not list:
                    raise TypeError
            if any([type(e) not in allowed for e in l]):
                raise TypeError
            l = [[self.html_all(ee) for ee in e] for e in l]
        except TypeError:
            if opts.get('orient')=='row':
                l = [[self.html_all(e) for e in l]]
            else:
                l = [[self.html_all(e)] for e in l]
        #finally:
        foo = set(t_opts).intersection(opts)
        nt_opts = dict((k, opts[k]) for k in foo)
        #nt_opts['stretch'] = not opts['outer']
        #opts['outer'] = False
        return to_table(l, **nt_opts)

    def html_array(self, np_array, opts):
        I = np_array
        if str(I.dtype).startswith('complex'):
            I = np.absolute(I)
        dt = I.dtype
        mx = np.max(I)
        mn = np.min(I)       
        I = I.astype('float')

        if I.ndim==2 or I.ndim==3:
            if I.shape[0]<50:
                I = grow(I)
            if opts.get('clip'):
                I = clip_arr(I, pct=clip)
            if opts.get('bound'):
                I = 255*(I-mn)/(mx-mn)
            im = Image.fromarray(I.astype('uint8'))
            self.all_options[id(im)]=opts
            s = self.html_pil(im, opts)
            #if str(dt).startswith('float'):
            #    l = ['min:{:.3f} max:{:.3f} {}'.format(mn,mx,dt), im]
            #if str(dt).startswith('int'):
            #    l = ['min:{} max:{} {}'.format(mn,mx,dt), im]
            #s = self.html_all(l, opts)
        elif I.ndim == 1:
            plt.clf()
            plt.plot(I)
            s = self.html_all(plt.gcf(), opts)
        return s

    def html_plot(self, fig, opts):
        bio = BytesIO()
        fig.dpi=DPI
        if opts.get('size'):
            w,h = opts.get('size')
            fig.set_size_inches(w/DPI,h/DPI)
        fig.savefig(bio, format='png', dpi=DPI)
        bio.seek(0)
        return self.html_image(bio)

    def html_pil(self, pil_image, opts):
        bio = BytesIO()
        if opts.get('size'):
            size = opts.get('size')
            pil_image = pil_image.resize(size)
        pil_image.save(bio, 'PNG')
        bio.seek(0)
        return self.html_image(bio)

    def html_image(self, bio):
        image_type = "png"  #data:image/png;base64
        image_encoded = b64encode(bio.read())
        image_tag = """
            <img style="display:block;" src="data:image/{1};base64,{0}">
            </img>""".format(urllib.quote(image_encoded), image_type) # width="100%" height="100%"
        return image_tag

    def html_audio(self, I):
        bio = BytesIO()
        wav.write(bio, 44100, I)
        bio.seek(0)
        audio_type = "wav"
        sound_encoded = b64encode(bio.read())
        sound_tag = """
            <audio controls src="data:audio/{1};base64,{0}">
            </audio>""".format(urllib.quote(sound_encoded), audio_type)
        return sound_tag


def clip_arr(z, pct=.99):
    return np.clip(z, 0., np.percentile(z, pct*100.))

def grow(I, m=10):
    w,h = I.shape
    II = np.empty((w*m,h*m), dtype=I.dtype)
    for i in range(w):
        for j in range(h):
            II[i*m:(i+1)*m,j*m:(j+1)*m]=I[i,j]
    return II

def fast_resample(I, newshape):
    x = np.linspace(0., 1., I.shape[0])
    y = I
    f = interp1d(x, y, kind='linear')
    newx = np.linspace(0., 1., newshape)
    return f(newx)




