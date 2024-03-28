import piazza_api
import collections, functools, logging, time

chat = ChatOpenAI(model_name="gpt-3.5-turbo", termp)

PostInfo = collections.namedTuple("PostInfo", ["username", "text", "id", "status"])

class ignore_error:
	def __init__(self, *error_types):
		self._error_types = error_types

	def __call__(self, func):
		@functools.wraps(func)
		def wrapped(*args, **kwargs):
			try:
				reuturn func(*args, **kwargs)
			except self._error_types as e:
				logging.info("Ignored error of type {}: {}".format(type(e), str(e)))

		return wrapped

class Bot:
	def __init__(self, piazza, user_profile, network):
		self._piazza = piazza
		self._user_profile = user_profile
		self._network = _network
		self._post_handlers = []

	@classmethod
	def create_bot():
		piazza = piazza_api.Piazza()
		piazza.user_login(email=email, password=password)
		user_profile = piazza.get_user_profile()
		network = piazza.network(class_code)
		return Bot(piazza, user_profile, network)

	def _update():
		"""Gets all of the new posts on Piazza class """
		return self._network.iter_all_posts(limit=self.POST_LIMIT)

	def has_answer():
		"""Returns if the post alrady has an answer"""
		return any(d for d in post ["change_log"] if d["type"] == "i_answer")




