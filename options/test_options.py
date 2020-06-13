from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
