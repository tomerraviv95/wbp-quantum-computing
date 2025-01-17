from python_code.decoders.bp.bp_decoder import BPDecoder
from python_code.decoders.hard_bp.hard_bp import HardBPDecoder
from python_code.decoders.hard_decision.hd_decoder import HDDecoder
from python_code.decoders.hard_wbp.hard_wbp import HardWBPDecoder
from python_code.decoders.wbp.wbp_decoder import WBPDecoder
from python_code.utils.constants import DecoderType

DECODERS_TYPE_DICT = {DecoderType.bp.name: BPDecoder,
                      DecoderType.wbp.name: WBPDecoder,
                      DecoderType.hd.name: HDDecoder,
                      DecoderType.hard_bp.name: HardBPDecoder,
                      DecoderType.hard_wbp.name: HardWBPDecoder}
