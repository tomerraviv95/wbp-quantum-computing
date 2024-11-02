from python_code.decoders.bp.bp_decoder import BPDecoder
from python_code.decoders.wbp.wbp_decoder import WBPDecoder
from python_code.utils.constants import DecoderType

DECODERS_TYPE_DICT = {DecoderType.bp.name: BPDecoder,
                      DecoderType.wbp.name: WBPDecoder}
