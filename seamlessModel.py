from transformers import AutoProcessor, SeamlessM4Tv2Model
# import torchaudio

# these are the languages we will try and use as well as workaround
# LANGUAGES = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

text_inputs = processor(text="Once upon a time, in the magical land of Rainbow Valley, there lived a unicorn named Sparkle. Sparkle wasn't like the other unicorns; her mane shimmered in all the colors of the rainbow, and her horn sparkled with a bright, twinkling light.", src_lang="eng", return_tensors="pt")
# audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
output_tokens = model.generate(**text_inputs, tgt_lang="afr", generate_speech=False)
translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

print("Translated text (Afrikaans):", translated_text_from_text)

is_approved = input("Is the translation approved? (y/n): ")

if is_approved.lower() == "y":
    print("translation approved!")
else:
    # this data set will be used to fine tune the model
    corrected_text = input("Please enter corrected translation:")
    print("Thank you for the correction!")