import nltk
import language_tool_python
from spellchecker import SpellChecker

class CreateArrays:
    def __init__(self):
        # install java to run language_tool_python if using jupyter hub
        """
        jdk.install('17')
        java_home = os.path.expanduser("~/.jdk/jdk-17.0.17+10")
        os.environ['JAVA_HOME'] = java_home
        os.environ['PATH'] = f"{os.environ['PATH']}:{java_home}/bin"
        """

        # download nltk resources
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

        # initialize grammar checker
        self.tool = language_tool_python.LanguageTool('en-US')

        # initialize spell checker
        self.spell = SpellChecker(language='en')