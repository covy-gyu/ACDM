from infer.POCR.pororo import Pororo
from infer.POCR.pororo.pororo import SUPPORTED_TASKS
import warnings

warnings.filterwarnings("ignore")

def init():
    pocr_obj = PororoOcr()
    return pocr_obj

class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result["description"]:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = ""

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path