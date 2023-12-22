""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# for portuguese
_pt_symbols = "\u0303"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
symbols += list(_pt_symbols)

# Special symbol ids
SPACE_ID = symbols.index(" ")

_ch_letters = 'abcdefghijklmnopqrstuvwxyzøŋœɑɕɛɨɪɵʂʈʌʏʐʑʒ'
_tone = "12345"
zh_symbols = [_pad] + list(_punctuation) + list(_tone) + list(_ch_letters)

_ja_letters = ':Nabcdefghijkmnopqrstuwyz'
ja_symbols = [_pad] + list(_punctuation) + list(_ja_letters)