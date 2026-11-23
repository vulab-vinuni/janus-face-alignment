# Nature-style restyling applied at savefig time (font, sizes, weights, line widths).
# Design size: 183 mm (7.2 in) double-column; text 5.5-7 pt, panel labels 8 pt bold, Arial-metric font.
import os, re, builtins
_orig_import = builtins.__import__
_done = {'p': False}

FONT = 'Liberation Sans'          # metric-compatible with Arial
PANEL = re.compile(r'^\(?[a-l]\)?$')

def _patch():
    import matplotlib, matplotlib.colors
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.legend import Legend
    orig = Figure.savefig

    def role_map(fig):
        roles = {}
        for ax in fig.axes:
            for t in [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]:
                roles[id(t)] = 'label'
            for t in ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'):
                roles[id(t)] = 'tick'
            for t in [ax.xaxis.offsetText, ax.yaxis.offsetText]:
                roles[id(t)] = 'tick'
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts(): roles[id(t)] = 'legend'
                if leg.get_title(): roles[id(leg.get_title())] = 'legend'
        for leg in fig.legends:
            for t in leg.get_texts(): roles[id(t)] = 'legend'
        return roles

    def restyle(fig, k):
        roles = role_map(fig)
        texts = [t for t in fig.findobj(Text) if t.get_text().strip()]
        lab = [getattr(t, '_norig', t.get_fontsize()) for t in texts if roles.get(id(t)) == 'label']
        ref = sorted(lab)[len(lab) // 2] if lab else 10.0
        for t in texts:
            if not hasattr(t, '_norig'):
                t._norig = t.get_fontsize(); t._nbold = t.get_fontweight() in ('bold', 'heavy', 'black', 700, 800, 900)
            s = t.get_text().strip(); r = roles.get(id(t))
            t.set_fontfamily([FONT, 'DejaVu Sans'])
            if '$' in s:
                u = t.get_text()
                for a_, b_ in (('\\mathbfit{', '\\mathit{'), ('\\mathbf{', '\\mathrm{'), ('\\boldsymbol{', '{'), ('\\bf ', '')):
                    u = u.replace(a_, b_)
                t.set_text(u); s = u.strip()
            try: t.set_math_fontfamily('custom')
            except Exception: pass
            if PANEL.match(s) and (t._nbold or s.startswith('(')) and r is None and sum(matplotlib.colors.to_rgb(t.get_color())) < 0.6:
                t.set_text(s.strip('()')); t.set_fontsize(8 * k); t.set_fontweight('bold'); continue
            t.set_fontweight('normal')
            if r == 'label': size = 7
            elif r == 'tick': size = 6.5
            elif r == 'legend': size = 6
            else:
                q = t._norig / ref
                size = 7 if q >= 0.95 else 6.5 if q >= 0.82 else 6 if q >= 0.68 else 5.5
            t.set_fontsize(size * k)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.collections import Collection
        def thin(v, f, lo, hi):
            p = v / k
            return min(max(p * f, lo), hi) * k if p > 0 else v
        for o in fig.findobj(Line2D):
            if not hasattr(o, '_nlw'): o._nlw = o.get_linewidth(); o._nmew = o.get_markeredgewidth()
            o.set_linewidth(thin(o._nlw, 0.6, 0.4, 1.0)); o.set_markeredgewidth(thin(o._nmew, 0.5, 0.25, 0.5))
        for o in fig.findobj(Collection):
            if not hasattr(o, '_nlw'): o._nlw = list(o.get_linewidths())
            o.set_linewidths([thin(v, 0.5, 0.25, 0.75) for v in o._nlw])
        for o in fig.findobj(Patch):
            if o in [sp for ax in fig.axes for sp in ax.spines.values()] or o is fig.patch: continue
            if not hasattr(o, '_nlw'): o._nlw = o.get_linewidth()
            o.set_linewidth(thin(o._nlw, 0.5, 0.25, 0.6))
        for ax in fig.axes:
            if getattr(ax, 'name', '') != 'rectilinear': continue
            vis = [sp.get_visible() for sp in ax.spines.values()]
            xl = any(t.get_visible() and t.get_text() for t in ax.get_xticklabels())
            yl = any(t.get_visible() and t.get_text() for t in ax.get_yticklabels())
            right = ax.yaxis.get_ticks_position() == 'right' or ax.yaxis.get_label_position() == 'right'
            top = ax.xaxis.get_label_position() == 'top'
            is_cbar = hasattr(ax, '_colorbar') or ax.get_label() == '<colorbar>'
            sib = [b for b in ax.get_shared_y_axes().get_siblings(ax) + ax.get_shared_x_axes().get_siblings(ax) if b is not ax]
            p0 = ax.get_position()
            row = [b for b in fig.axes if b is not ax and abs(b.get_position().y0 - p0.y0) < 0.01 and abs(b.get_position().height - p0.height) < 0.01]
            sib = sib + row
            sl = any(any(t.get_visible() and t.get_text() for t in b.get_yticklabels()) for b in sib) and len(ax.get_yticks()) > 0 and ax.yaxis.get_visible()
            if all(vis) and (xl or yl or sl) and not is_cbar:
                ax.spines['top'].set_visible(top); ax.spines['right'].set_visible(right)
                if not top: ax.spines['top'].set_visible(False)
                if not right: ax.spines['right'].set_visible(False)
                ax.tick_params(which='both', direction='out', top=top, right=right,
                               bottom=not top, left=not right)
                ax.minorticks_off()
            elif all(vis) and not is_cbar:
                ax.tick_params(which='both', direction='out', top=False, right=False)
                ax.minorticks_off()
            ax.tick_params(which='major', labelsize=6.5 * k, width=0.5 * k, length=2.5 * k, pad=2 * k, labelfontfamily=FONT)
            for sp in ax.spines.values(): sp.set_linewidth(0.5 * k)

    def savefig(self, *a, **kw):
        k = self.get_size_inches()[0] / 7.2
        rc = {'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic',
              'mathtext.bf': FONT + ':bold', 'mathtext.sf': FONT, 'font.family': FONT, 'font.weight': 'normal',
              'axes.labelweight': 'normal', 'axes.titleweight': 'normal'}
        with matplotlib.rc_context(rc):
            if not getattr(self, '_nstyled', False):
                restyle(self, k)
                try: self.canvas.draw()
                except Exception: pass
                if kw.get('bbox_inches') == 'tight' or matplotlib.rcParams.get('savefig.bbox') == 'tight':
                    try:
                        bb = self.get_tightbbox(self.canvas.get_renderer())
                        k = (bb.width + 2 * kw.get('pad_inches', 0.1)) / 7.2
                        restyle(self, k); self.canvas.draw()
                        bb = self.get_tightbbox(self.canvas.get_renderer())
                        k = (bb.width + 2 * kw.get('pad_inches', 0.1)) / 7.2
                    except Exception as e: print('nstyle tight', e)
                restyle(self, k); self._nstyled = True
            return orig(self, *a, **kw)
    Figure.savefig = savefig
    import matplotlib.pyplot as plt
    def psave(*a, **kw): return plt.gcf().savefig(*a, **kw)
    plt.savefig = psave

def _hook(name, *a, **kw):
    m = _orig_import(name, *a, **kw)
    if not _done['p'] and name.startswith('matplotlib'):
        import sys
        pp = sys.modules.get('matplotlib.pyplot')
        if pp is not None and hasattr(pp, 'nipy_spectral') and hasattr(sys.modules['matplotlib.figure'], 'Figure'):
            _done['p'] = True
            try: _patch()
            except Exception as e: print('nstyle patch failed', e)
    return m
builtins.__import__ = _hook
