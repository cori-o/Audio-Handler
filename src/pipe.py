
class BasePipeline:
    def __init__(self, config):
        self.config = config
    
    def set_env(self):
        pass 
    

class SpeechActivityDetector(BasePipeline):
    def __init__(self, config):
        super().__init__(config)


class SegmentManager(BasePipeline):
    def __init__(self, config)
        super().__init__(config)

