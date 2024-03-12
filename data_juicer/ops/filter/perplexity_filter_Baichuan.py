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

OP_NAME = 'perplexity_filter_Baichuan'

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
                 max_sequence_length = None,
                 stride = 512,
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
        # self.tokenizer = AutoTokenizer.from_pretrained("/mnt/cache/share_data/lishiyu/models/Baichuan2-7B-Base", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan2-7b-base", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("baichuan2-7b-base",  trust_remote_code=True).eval()

        # self.model = AutoModelForCausalLM.from_pretrained("/mnt/cache/share_data/lishiyu/models/Baichuan2-7B-Base", device_map="auto", cache_dir = '/mnt/lustre/xieshuo/lustrenew/workspace/dj_mixture_challenge/toolkit/data-juicer/cache', trust_remote_code=True).eval()
        self.max_sequence_length = max_sequence_length
        self.stride = stride

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
            tokenizer = self.tokenizer
            model = self.model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model_input = sample['instruction']+sample['input']+sample['output']
            model_input = model_input[:min(2048, len(model_input))]
            max_sequence_length = self.max_sequence_length
            stride = self.stride
            # model_input = model_input[:1200]

            if max_sequence_length is not None:
            # Use the provided method with max_sequence_length
                stride = stride
                max_length = max_sequence_length
                seq_len = len(model_input)

                nlls = []
                prev_end_loc = 0
                for begin_loc in tqdm(range(0, seq_len, stride)):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                    input_encode = tokenizer.encode(model_input[begin_loc:end_loc], return_tensors='pt').to(device)
                    target_encode = input_encode.clone()
                    target_encode[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = model(input_encode, labels=target_encode)
                        neg_log_likelihood = outputs.loss

                    nlls.append(neg_log_likelihood)

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

                ppl = torch.exp(torch.stack(nlls).mean())
            else:
                input_encode = tokenizer.encode(model_input,return_tensors='pt').to(device)
                target_encode = input_encode.clone()

                with torch.no_grad():
                    outputs = model(input_encode, labels=target_encode)
                    neg_log_likelihood = outputs.loss
                    ppl = torch.exp(neg_log_likelihood)
                    own_ppl = ppl.item()

        sample[Fields.stats][StatsKeys.perplexity] = own_ppl

        return sample

    def process(self, sample):
        return sample[Fields.stats][StatsKeys.perplexity] <= self.max_ppl
 