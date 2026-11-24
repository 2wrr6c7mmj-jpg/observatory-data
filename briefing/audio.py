"""音声：VOICEVOX コア（Python バインディング）で文ごとに合成し、つないで mp3 にする。開始時刻は文ごとに記録する。

声は briefing/voice.json の style_id 1 つで決まる（差し替えは style_id と vvm を一緒に）。
用語の読みは briefing/readings.json をユーザー辞書として入れる。取得物（whl・onnxruntime・辞書・声のモデル）は
tools/voicevox_setup.sh が版と SHA-256 を確かめて置く。
WAV のつなぎは標準ライブラリ（wave）、mp3 は lameenc（版とハッシュを固定）。
"""
import glob
import io
import json
import re
import os
import wave

SAMPLE_PAUSE = {'line': 0.30, 'event': 0.90}
INTRO_GAP = 0.8


def _here(name):
    return os.path.join(os.path.dirname(__file__), name)


def load_voice():
    with open(_here('voice.json'), encoding='utf-8') as f:
        return json.load(f)


def load_readings():
    with open(_here('readings.json'), encoding='utf-8') as f:
        return [tuple(w) for w in json.load(f)['words']]


class Pcm:
    """16 ビット・モノラルの PCM をためる。"""

    def __init__(self, rate=24000):
        self.rate, self.buf = rate, bytearray()

    @property
    def seconds(self):
        return len(self.buf) / 2 / self.rate

    def add_wav(self, data):
        with wave.open(io.BytesIO(data)) as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError('expected 16-bit mono wav')
            if not self.buf:
                self.rate = w.getframerate()
            elif w.getframerate() != self.rate:
                raise ValueError('sample rate changed')
            self.buf += w.readframes(w.getnframes())

    def add_silence(self, sec):
        self.buf += b'\x00\x00' * int(self.rate * sec)


def assemble(items, synth_fn, pcm=None):
    """items = [(キー, 文, 文のあと無音の秒)…] を順に合成してつなぐ。{キー: 開始秒}（2 桁）と Pcm を返す。
    synth_fn(文) -> WAV のバイト列（試験では差し替える）。"""
    pcm = pcm or Pcm()
    starts = {}
    for key, text, gap in items:
        starts[key] = round(pcm.seconds, 2)
        pcm.add_wav(synth_fn(text))
        pcm.add_silence(gap)
    return starts, pcm


def to_mp3(pcm, bitrate=64):
    import lameenc  # 遅い import
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(pcm.rate)
    enc.set_channels(1)
    enc.set_quality(2)
    return bytes(enc.encode(bytes(pcm.buf))) + bytes(enc.flush())


# 声の設定（voice.json の "tune"）：話す速さ・抑揚の強さ・高さ・文の前後の無音。範囲の外の値は使わない。
TUNE = {'speed': 'speed_scale', 'intonation': 'intonation_scale', 'pitch': 'pitch_scale', 'pre': 'pre_phoneme_length', 'post': 'post_phoneme_length'}
TUNE_RANGE = {'speed': (0.7, 1.4), 'intonation': (0.3, 1.6), 'pitch': (-0.15, 0.15), 'pre': (0.0, 0.5), 'post': (0.0, 0.5)}
KANA_SCRIPT = re.compile(r"[ァ-ヶー'/、_？]+")   # VOICEVOX の AquesTalk 風記法に出る字


def bare(kana):
    """調声のカナから記号（下がり目 '・区切り /・間 、・無声化 _・疑問 ？）を除いた、読みだけ。"""
    return re.sub(r"['/、_？]", '', kana or '')


def spoken(text):
    """読み上げに渡す形：かぎ括弧を外す（「神舟24号」の、が「、しんしゅう…、の」と間のあいた読みになる）。字幕の文はそのまま。"""
    return text.replace('「', '').replace('」', '').replace('『', '').replace('』', '')


# 文末の落ち着かせ（voice.json の "ending"）。合成器は文によって、文末の句を高く持ち上げたまま終えたり、最後の拍を
# 下げきらなかったりする（聞くと、文末が上がったり下がったりして聞こえる）。どの文も同じ形で終わるように寄せる。
# 値は高さ（VOICEVOX の pitch、対数）の差で、低いところ（声の出る拍を低い順に並べた 1 割目）から測る。
#   peak：文末の句の山の上限（〜ました。）   open：核のあとに声の出る拍が無い文末（〜ます。）の山の上限
#   end ：核のあとの最後の拍（〜ました。の「た」）の上限
ENDING_RANGE = {'peak': (0.05, 0.40), 'open': (0.02, 0.30), 'end': (-0.05, 0.15)}


def ending_of(conf):
    """voice.json の "ending" から、使える値だけ。3 つそろわなければ None（何もしない）。"""
    out = {}
    for key, (lo, hi) in ENDING_RANGE.items():
        v = (conf or {}).get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
            out[key] = float(v)
    return out if len(out) == len(ENDING_RANGE) else None


def low_pitch(pitches):
    """声の出る拍の高さを低い順に並べた 1 割目（その声の「低いところ」）。拍が少なければ None。"""
    voiced = sorted(p for p in pitches if p > 0)
    return voiced[len(voiced) // 10] if len(voiced) >= 4 else None


def settle_ending(phrases, peak, open, end, base=None):
    """phrases（AudioQuery の accent_phrases）の文末を書き換える。下げるだけで、上げない。疑問の文は触らない。
    1. 文末の句の山が上限を超えていれば、文末の句と（間を置かずに続く）その前の句を、山へ向かって少しずつ下げる。
       句の中の形（どこで上がり、どこで下がるか）は変えない。
    2. 核のあとの最後の拍が上限より高ければ、核の直後からその拍へ向かって下げる。
    base は低いところ（省けばその文から）。返すのは (山を下げた量, 最後の拍を下げた量)。"""
    if base is None:
        base = low_pitch(m.pitch for ap in phrases for m in ap.moras)
    if base is None or not phrases or phrases[-1].is_interrogative:
        return 0.0, 0.0
    last = phrases[-1].moras
    vi = [i for i, m in enumerate(last) if m.pitch > 0]
    if not vi:
        return 0.0, 0.0
    after = [i for i in vi if i >= phrases[-1].accent]   # 核のあとの、声の出る拍
    floor = base + end
    top = max(vi, key=lambda i: last[i].pitch)
    down = max(0.0, last[top].pitch - (base + (peak if after else open)))
    if down:
        lead = list(phrases[-2].moras) if len(phrases) > 1 and not phrases[-2].pause_mora else []
        ramp = lead + list(last[:top + 1])
        for k, m in enumerate(ramp, 1):
            if m.pitch > 0:
                m.pitch -= down * k / len(ramp)
        for i in vi:
            if i > top:   # 山のあと：同じだけ下げる。核のあとの拍は、もう floor より低ければそこで止める
                p = last[i].pitch
                last[i].pitch = max(p - down, min(p, floor)) if i in after else p - down
    drop = 0.0
    if after:
        first, tail = phrases[-1].accent, after[-1]
        drop = max(0.0, last[tail].pitch - floor)
        for i in after:
            last[i].pitch -= drop * (i - first + 1) / (tail - first + 1)
    return down, drop


class Voice:
    """VOICEVOX コアの合成器。root は tools/voicevox_setup.sh が作ったディレクトリ（onnxruntime/・dict/・vvm/）。"""

    def __init__(self, root, readings=None, voice=None):
        from voicevox_core import UserDictWord  # 遅い import
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, UserDict, VoiceModelFile
        self.voice = voice or load_voice()
        ort = glob.glob(f'{root}/onnxruntime/*/lib/' + Onnxruntime.LIB_RECOMMENDED_VERSIONED_FILENAME)[0]
        self.synth = Synthesizer(Onnxruntime.load_once(filename=ort), OpenJtalk(glob.glob(f'{root}/dict/open_jtalk_dic_utf_8-*')[0]))
        words = readings if readings is not None else load_readings()
        self.skipped = []   # 辞書に入れられなかった読み（アクセント核が拍の数を超える など）
        if words:
            d = UserDict()
            for surface, pron, accent in words:
                try:
                    d.add_word(UserDictWord(surface, pron, int(accent), priority=8))
                except Exception as e:   # 1 語の誤りで全体を止めない（その日の調声の語が入る）
                    self.skipped.append(f'{surface}（{type(e).__name__}）')
            self.synth.open_jtalk.use_user_dict(d)
        with VoiceModelFile.open(f"{root}/vvm/{self.voice['vvm']}") as m:
            ids = {st.id for ch in m.metas for st in ch.styles}
            if self.voice['style_id'] not in ids:
                raise ValueError(f"style {self.voice['style_id']} is not in {self.voice['vvm']}")
            self.synth.load_voice_model(m)
        self.style = self.voice['style_id']
        self.tune = dict(self.voice.get('tune') or {})
        self.ending = ending_of(self.voice.get('ending'))
        self.base = None    # 文末の落ち着かせの基準（その回の全部の文の低いところ）。calibrate が入れる
        self.untuned = []   # 調声のカナが使えず、文から読んだ文

    def kana(self, text):
        return self.synth.create_audio_query(spoken(text), self.style).kana

    def tuned(self, text, kana):
        """調声のカナ（アクセントの下がり目・句の区切り・間を直したもの）が使えるか。使えればそのカナ、だめなら None。
        使えるのは、記法として読めて、記号を除いた読みが自動の読みと同じとき（字幕と違うことを読ませない。読みそのものを
        変えるのは readings）。"""
        if not kana:
            return None
        try:
            self.synth.create_audio_query_from_kana(kana, self.style)
        except Exception:
            return None
        return kana if bare(kana) == bare(self.kana(text)) else None

    def query(self, text, kana=None):
        """1 文の読み方（AudioQuery）。kana（調声のカナ）が使えればそれで、だめなら文から。"""
        use = self.tuned(text, kana)
        return self.synth.create_audio_query_from_kana(use, self.style) if use else self.synth.create_audio_query(spoken(text), self.style)

    def calibrate(self, pairs):
        """pairs = [(文, 調声のカナ または None)…]：その回の全部の文から、声の低いところを決める（文末の高さを文どうしでそろえる）。"""
        if self.ending:
            self.base = low_pitch(m.pitch for text, kana in pairs for ap in self.query(text, kana).accent_phrases for m in ap.moras)

    def wav(self, text, kana=None):
        """1 文の音声。声の設定（voice.json の tune）と、文末の落ち着かせ（ending）を当てる。"""
        if kana and not self.tuned(text, kana):
            self.untuned.append(text)
        q = self.query(text, kana)
        for key, attr in TUNE.items():
            v = self.tune.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and TUNE_RANGE[key][0] <= v <= TUNE_RANGE[key][1]:
                setattr(q, attr, float(v))
        if self.ending:
            settle_ending(q.accent_phrases, base=self.base, **self.ending)
        return self.synth.synthesis(q, self.style)
