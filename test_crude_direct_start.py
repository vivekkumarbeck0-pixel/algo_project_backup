from unittest.mock import Mock, patch

import crude_ppo_paper_runner
import trading_crude


def test_direct_start_runs_crude_ppo_and_health_server():
    thread = Mock()
    with patch.object(trading_crude.threading, "Thread", return_value=thread) as thread_cls, \
         patch.object(trading_crude, "start_ppo_paper_thread") as start_ppo, \
         patch.object(trading_crude.app, "run") as run_server, \
         patch.dict("os.environ", {"PORT": "12345"}):
        trading_crude.main()

    thread_cls.assert_called_once_with(
        target=trading_crude.start_trading_engine, name="crude-engine", daemon=True
    )
    thread.start.assert_called_once_with()
    start_ppo.assert_called_once_with()
    run_server.assert_called_once_with(host="0.0.0.0", port=12345)


def test_ppo_thread_can_be_disabled():
    with patch.dict("os.environ", {"ENABLE_PPO_PAPER": "0"}), \
         patch.object(trading_crude.threading, "Thread") as thread_cls:
        assert trading_crude.start_ppo_paper_thread() is None

    thread_cls.assert_not_called()


def test_ppo_thread_is_enabled_by_default():
    thread = Mock()
    with patch.dict("os.environ", {"ENABLE_PPO_PAPER": ""}), \
         patch.object(trading_crude.threading, "Thread", return_value=thread) as thread_cls:
        assert trading_crude.start_ppo_paper_thread() is thread

    thread_cls.assert_called_once_with(
        target=crude_ppo_paper_runner.run_live_forever,
        name="ppo-paper-runner",
        daemon=True,
    )
    thread.start.assert_called_once_with()
