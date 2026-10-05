"""Give-way points inside long tools: ``check()`` raises if the engine is
rendering full detail and the app has asked for something newer (see
Engine._checkpoint). Outside such a render it does nothing."""

check = lambda: None  # noqa: E731  (set by the engine)
