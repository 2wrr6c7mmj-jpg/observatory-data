"""音声：VOICEVOX コア（Python バインディング）で文ごとに合成し、つないで mp3 にする。開始時刻は文ごとに記録する。

声は briefing/voice.json の style_id 1 つで決まる（差し替えは style_id と vvm を一緒に）。
用語の読みは briefing/readings.json をユーザー辞書として入れる。取得物（whl・onnxruntime・辞書・声のモデル）は
tools/voicevox_setup.sh が版と SHA-256 を確かめて置く。
WAV のつなぎは標準ライブラリ（wave）、mp3 は lameenc（版とハッシュを固定）。
"""
import glob
import io
import json
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


class Voice:
    """VOICEVOX コアの合成器。root は tools/voicevox_setup.sh が作ったディレクトリ（onnxruntime/・dict/・vvm/）。"""

    def __init__(self, root, readings=None, voice=None):
        from voicevox_core import UserDictWord  # 遅い import
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, UserDict, VoiceModelFile
        self.voice = voice or load_voice()
        ort = glob.glob(f'{root}/onnxruntime/*/lib/' + Onnxruntime.LIB_RECOMMENDED_VERSIONED_FILENAME)[0]
        self.synth = Synthesizer(Onnxruntime.load_once(filename=ort), OpenJtalk(glob.glob(f'{root}/dict/open_jtalk_dic_utf_8-*')[0]))
        words = readings if readings is not None else load_readings()
        if words:
            d = UserDict()
            for surface, pron, accent in words:
                d.add_word(UserDictWord(surface, pron, int(accent), priority=8))
            self.synth.open_jtalk.use_user_dict(d)
        with VoiceModelFile.open(f"{root}/vvm/{self.voice['vvm']}") as m:
            ids = {st.id for ch in m.metas for st in ch.styles}
            if self.voice['style_id'] not in ids:
                raise ValueError(f"style {self.voice['style_id']} is not in {self.voice['vvm']}")
            self.synth.load_voice_model(m)
        self.style = self.voice['style_id']

    def kana(self, text):
        return self.synth.create_audio_query(text, self.style).kana

    def wav(self, text):
        return self.synth.tts(text, self.style)
