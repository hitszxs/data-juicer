# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import PositiveFloat
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model
import torch
from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document
from ..op_fusion import INTER_WORDS

OP_NAME = 'perplexity_filter'

with AvailabilityChecking(['sentencepiece', 'kenlm'], OP_NAME):
    import kenlm  # noqa: F401
    import sentencepiece  # noqa: F401


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class PerplexityFilter(Filter):
    """Filter to keep samples with perplexity score less than a specific max
    value."""

    def __init__(self,
                 lang: str = 'en',
                 max_ppl: PositiveFloat = 1500,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: Compute perplexity for samples in which language.
        :param max_ppl: The max filter perplexity in this op, samples
            will be filtered if their perplexity exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.max_ppl = max_ppl
        self.lang = lang
        self.sp_model_key = prepare_model(lang=lang,
                                          model_type='sentencepiece')
        # self.kl_model_key = prepare_model(lang=lang, model_type='kenlm')
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)

        # self.tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/xieshuo/lustrenew/workspace/dj_mixture_challenge/toolkit/data-juicer/cache/models--Qwen--Qwen-14B/snapshots/c4051215126d906ac22bb67fe5edb39a921cd831", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B", device_map="auto").eval()


    def compute_stats(self, sample, context=False):
        # torch.multiprocessing.set_start_method('spawn')

        # check if it's computed already
        if StatsKeys.perplexity in sample[Fields.stats]:
            return sample

        # tokenization
        words_key = f'{InterVars.words}-{self.sp_model_key}'
        if context and words_key in sample[Fields.context]:
            words = sample[Fields.context][words_key]
        else:
            # import pdb;pdb.set_trace()
            # tokenizer = get_model(self.sp_model_key, self.lang,
                                #   'sentencepiece')
            # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
            tokenizer = self.tokenizer
            model = self.model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # if torch.cuda.device_count() > 1:
                # print("Let's use", torch.cuda.device_count(), "GPUs!")
                # model = nn.DataParallel(model)
            model = model.to(device)
            model_input = sample['instruction']+sample['input']+sample['output']
            model_input = model_input[:1200]
            input_encode = tokenizer.encode(model_input,return_tensors='pt').to(device)
            # attention_mask_encode = tokenizer.encode(sample[''],return_tensors='pt').to(device)
            target_encode = input_encode.clone()
            # output_encode = tokenizer.encode(sample['output'],return_tensors='pt').to(device)
            # words = get_words_from_document(
            #     sample[self.text_key],
            #     token_func=tokenizer.encode if tokenizer else None)
            # if context:
            #     sample[Fields.context][words_key] = words
            
        # text = ' '.join(words)

        # text = ' '.join(map(str, words))
        # compute perplexity
        # logits, length = 0, 0
        # kenlm_model = get_model(self.kl_model_key, self.lang, 'kenlm')
        
        with torch.no_grad():
            outputs = model(input_encode, labels=target_encode)
            neg_log_likelihood = outputs.loss
            ppl = torch.exp(neg_log_likelihood)
            own_ppl = ppl.item()
        # for line in text.splitlines():
        #     logits += kenlm_model.score(line)
        #     length += (len(line.split()) + 1)
        # ppl = (10.0**(-logits / length)) if length != 0 else 0.0
        sample[Fields.stats][StatsKeys.perplexity] = own_ppl

        return sample

    def process(self, sample):
        return sample[Fields.stats][StatsKeys.perplexity] <= self.max_ppl
 