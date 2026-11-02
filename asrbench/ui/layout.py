"""Layout Gradio: costruzione blocchi, wiring, callback di lingua e sidebar."""
from __future__ import annotations

import gradio as gr

from asrbench.core import state
from asrbench.core.logbuffer import clear as log_clear, event, path as log_path
from asrbench.core.logbuffer import tail as log_tail
from asrbench.engine.helpers import model_choices, short
from asrbench.engine.service import preload, unload_model
from asrbench.engine.trials import last_card_view, live_toggle, live_transcribe, reset_session
from asrbench.engine.trials import table_view
from asrbench.ui.i18n import TABLE_HEADERS, T, flow_html, model_info
from asrbench.ui.status import live_btn_state, side_update, status_html

CSS = """
#asr-sidebar { min-width: 300px; }
#asr-sidebar .prose table { font-size: 12px; }
#asr-drag { position: absolute; top: 64px; bottom: 0; left: 0; width: 10px;
  cursor: ew-resize; z-index: 1002; display: flex; align-items: center; justify-content: center; }
#asr-drag::after { content: ""; width: 3px; height: 48px; border-radius: 2px;
  background: var(--border-color-primary); }
#asr-drag:hover::after { background: var(--body-text-color); }
h2, h3 { margin-top: 6px !important; margin-bottom: 4px !important; }
"""

SIDEBAR_JS = """
<script>
(function () {
  window.asrSideW = window.asrSideW || 380;
  window.asrToggleSidebar = function () {
    var bar = document.getElementById('asr-sidebar');
    if (!bar) return false;
    var btn = bar.querySelector('.toggle-button');
    if (btn) { btn.click(); return true; }
    bar.classList.toggle('open');
    return true;
  };
  function applyW(bar, w) {
    w = Math.max(300, Math.min(Math.floor(window.innerWidth * 0.65), w));
    window.asrSideW = w;
    bar.style.width = w + 'px';
    bar.style.right = 'calc(' + w + 'px * -1)';
  }
  function install(bar) {
    if (!bar || bar.dataset.asrDrag) return;
    bar.dataset.asrDrag = '1';
    applyW(bar, window.asrSideW);
    var h = document.createElement('div');
    h.id = 'asr-drag';
    h.title = 'Trascina per ridimensionare il pannello';
    h.addEventListener('mousedown', function (e) {
      e.preventDefault();
      var sx = e.clientX, sw = bar.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      function mv(ev) { applyW(bar, sw + (sx - ev.clientX)); }
      function up() {
        document.removeEventListener('mousemove', mv);
        document.removeEventListener('mouseup', up);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
      document.addEventListener('mousemove', mv);
      document.addEventListener('mouseup', up);
    });
    bar.appendChild(h);
  }
  function scan() { install(document.getElementById('asr-sidebar')); }
  new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
  window.addEventListener('resize', function () {
    var bar = document.getElementById('asr-sidebar');
    if (bar) applyW(bar, window.asrSideW);
  });
  scan();
})();
</script>
"""


def on_select(model_id: str, lang: str = "it"):
    """Cambio dropdown: aggiorna scheda, diagramma flusso, stato, log, bottone live."""
    prev = state.SELECTED.get("model")
    state.SELECTED["model"] = model_id
    event("ui", f"select {short(prev)} -> {short(model_id)} (attivo={short(state.CURRENT.get('id'))})")
    return (model_info(model_id, lang), flow_html(model_id, lang),
            status_html(lang), log_tail(60), live_btn_state(lang))


def set_verbose(on: bool, lang: str = "it"):
    """Accende/spegne il log dei chunk live dalla sidebar."""
    state.VERBOSE["live_chunks"] = bool(on)
    event("ui", f"sidebar: log chunk live {'ON' if on else 'OFF'}")
    return status_html(lang), log_tail(60)


def ui_clear(lang: str = "it"):
    """Pulisce il buffer log mostrato in sidebar. Il file resta."""
    log_clear()
    event("ui", "sidebar: log pulito da utente")
    return status_html(lang), log_tail(60)


def set_lang(lang: str):
    """Flip IT<->EN e ri-renderizza tutte le stringhe UI nella nuova lingua."""
    lang = "en" if lang == "it" else "it"
    sel = state.SELECTED.get("model")
    return (lang,
            gr.update(value=T(lang, "title")),
            gr.update(value=T(lang, "guide")),
            gr.update(value=T(lang, "panel")),
            gr.update(value="IT" if lang == "en" else "EN"),
            gr.update(value=T(lang, "side_title")),
            gr.update(value=status_html(lang)),
            gr.update(label=T(lang, "log_label"), value=log_tail(30)),
            gr.update(label=T(lang, "verbose")),
            gr.update(value=T(lang, "refresh")),
            gr.update(value=T(lang, "clear")),
            gr.update(label="Log"),
            gr.update(value=T(lang, "s1")),
            gr.update(label=T(lang, "gt_label"), placeholder=T(lang, "gt_placeholder"),
                       info=T(lang, "gt_info")),
            gr.update(label=T(lang, "model_label")),
            gr.update(value=T(lang, "attiva")),
            gr.update(value=T(lang, "scarica")),
            gr.update(value=T(lang, "no_model")),
            gr.update(value=T(lang, "s2")),
            gr.update(label=T(lang, "acc_details")),
            gr.update(value=model_info(sel, lang)),
            gr.update(label=T(lang, "lat_label")),
            gr.update(value=T(lang, "flow_title")),
            gr.update(value=flow_html(sel, lang)),
            gr.update(value=T(lang, "s_trial")),
            gr.update(label=T(lang, "mic_label")),
            live_btn_state(lang),
            gr.update(value=T(lang, "s_live")),
            gr.update(label=T(lang, "live_text")),
            gr.update(label=T(lang, "live_status")),
            gr.update(value=T(lang, "s_session")),
            gr.update(headers=TABLE_HEADERS[lang], label=T(lang, "table_label")),
            gr.update(value=last_card_view(lang)),
            gr.update(value=T(lang, "reset")))


def build_demo():
    """Costruisce i blocchi Gradio e collega i callback. Ritorna demo."""
    with gr.Blocks(title="asr-bench", css=CSS) as demo:
        lang_state = gr.State(value="it")
        gr.HTML(SIDEBAR_JS)
        with gr.Row():
            title_md = gr.Markdown(T("it", "title"), elem_id="asr-title")
            lang_btn = gr.Button("EN", size="sm", scale=0, min_width=80)
            panel_btn = gr.Button(T("it", "panel"), size="sm", scale=0, min_width=120)
        guide_md = gr.Markdown(T("it", "guide"))
        panel_btn.click(None, js="() => { window.asrToggleSidebar(); }")
        with gr.Sidebar(label="Stato e log", open=True, position="right", elem_id="asr-sidebar", width=380):
            side_title = gr.Markdown(T("it", "side_title"))
            side_models = gr.HTML(status_html("it"))
            side_log = gr.Textbox(label=T("it", "log_label"), lines=10, max_lines=20,
                                  value=log_tail(30), info=f"{log_path()}")
            verbose_box = gr.Checkbox(label=T("it", "verbose"), value=False)
            with gr.Row():
                side_refresh = gr.Button(T("it", "refresh"), size="sm")
                side_clear = gr.Button(T("it", "clear"), size="sm")
                side_dl = gr.DownloadButton(label="Log", value=log_path(), size="sm")
            side_refresh.click(side_update, inputs=[lang_state], outputs=[side_models, side_log])
            side_clear.click(ui_clear, inputs=[lang_state], outputs=[side_models, side_log])
            verbose_box.change(set_verbose, inputs=[verbose_box, lang_state], outputs=[side_models, side_log])
            side_timer = gr.Timer(2.0)
            side_timer.tick(side_update, inputs=[lang_state], outputs=[side_models, side_log])

        with gr.Row():
            with gr.Column():
                s1_title = gr.Markdown(T("it", "s1"))
                gt_box = gr.Textbox(label=T("it", "gt_label"),
                                    placeholder=T("it", "gt_placeholder"),
                                    lines=1,
                                    info=T("it", "gt_info"))
                model_id = gr.Dropdown(choices=model_choices(), value=model_choices()[0][1],
                                       label=T("it", "model_label"))
                with gr.Row():
                    attiva_btn = gr.Button(T("it", "attiva"), variant="primary", size="sm")
                    scarica_btn = gr.Button(T("it", "scarica"), size="sm")
                attiva_status = gr.Markdown(T("it", "no_model"))
            with gr.Column():
                s2_title = gr.Markdown(T("it", "s2"))
                with gr.Accordion(T("it", "acc_details"), open=False) as acc_details:
                    info = gr.Markdown(model_info(model_choices()[0][1], "it"))
                with gr.Row():
                    lat_kpi = gr.Textbox(label=T("it", "lat_label"), value=state.DASH, lines=1, interactive=False)
                    rtf_kpi = gr.Textbox(label="RTF", value=state.DASH, lines=1, interactive=False)
                    wer_kpi = gr.Textbox(label="WER", value=state.DASH, lines=1, interactive=False)

        with gr.Row():
            flow_title = gr.Markdown(T("it", "flow_title"))
        with gr.Row():
            flow = gr.HTML(flow_html(model_choices()[0][1], "it"))

        with gr.Row():
            with gr.Column():
                s_trial = gr.Markdown(T("it", "s_trial"))
                live_audio = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                      label=T("it", "mic_label"))
                live_btn = gr.Button(T("it", "live_need", m=short(model_choices()[0][1])),
                                     variant="secondary", interactive=False)
            with gr.Column():
                s_live = gr.Markdown(T("it", "s_live"))
                live_text = gr.Textbox(label=T("it", "live_text"), lines=4)
                live_status = gr.Textbox(label=T("it", "live_status"), lines=1)
        trial_state = gr.State(value=None)

        s_session = gr.Markdown(T("it", "s_session"))
        with gr.Row():
            session_table = gr.Dataframe(headers=TABLE_HEADERS["it"], value=[], interactive=False,
                                         label=T("it", "table_label"), max_height=220)
            last_card = gr.Markdown(T("it", "no_session"))
        with gr.Row():
            dl_csv = gr.DownloadButton(label="CSV", value=state.CSV_PATH, size="sm")
            dl_jsonl = gr.DownloadButton(label="JSONL", value=state.JSONL_PATH, size="sm")
            reset_btn = gr.Button(T("it", "reset"), size="sm")

        lang_btn.click(set_lang, inputs=[lang_state],
                       outputs=[lang_state, title_md, guide_md, panel_btn, lang_btn,
                                side_title, side_models, side_log, verbose_box,
                                side_refresh, side_clear, side_dl,
                                s1_title, gt_box, model_id, attiva_btn, scarica_btn,
                                attiva_status, s2_title, acc_details, info, lat_kpi,
                                flow_title, flow, s_trial, live_audio, live_btn,
                                s_live, live_text, live_status, s_session,
                                session_table, last_card, reset_btn])
        model_id.change(on_select, inputs=[model_id, lang_state],
                        outputs=[info, flow, side_models, side_log, live_btn])
        attiva_btn.click(preload, inputs=[model_id, lang_state],
                         outputs=[attiva_status, side_models, side_log, live_btn])
        scarica_btn.click(unload_model, inputs=[lang_state],
                          outputs=[attiva_status, side_models, side_log, live_btn])
        live_btn.click(live_toggle, inputs=[model_id, gt_box, trial_state, lang_state],
                       outputs=[live_btn, gt_box, model_id, live_text, live_status,
                                session_table, last_card, lat_kpi, rtf_kpi, wer_kpi])
        live_audio.stream(live_transcribe, inputs=[model_id, live_audio, trial_state, lang_state],
                          outputs=[trial_state, live_text, live_status],
                          stream_every=1.0, time_limit=300)
        reset_btn.click(reset_session, inputs=[lang_state],
                        outputs=[session_table, last_card, lat_kpi, rtf_kpi, wer_kpi,
                                 side_models, side_log])
    return demo
