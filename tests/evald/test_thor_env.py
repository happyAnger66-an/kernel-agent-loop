"""模块: kernel_agent_loop.evald.thor_env
功能: Thor 会话准备在缺少 nvpmodel/jetson_clocks 时跳过而不抛错。
"""

from kernel_agent_loop.evald.thor_env import prepare_thor_session


# 用例: 开发机没有 Jetson 工具时返回跳过说明
# 参数: 无
def test_prepare_thor_session_skips_missing_tools():
    notes = prepare_thor_session()
    assert any("nvpmodel" in note for note in notes)
    assert any("jetson_clocks" in note for note in notes)
