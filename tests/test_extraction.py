r"""
FactExtractor（事实抽取器）的验收测试
================================================================

这些测试现在**全部是失败的**。让它们变绿，就是你的第 6 周作业。

运行：
    # 全部
    .\.venv\Scripts\python.exe -m pytest tests/test_extraction.py -v

    # 只跑核心的解析健壮性那一组（建议先集中攻这一组）
    .\.venv\Scripts\python.exe -m pytest tests/test_extraction.py -k "parse" -v

--------------------------------------------------------------------
这个测试文件教你的四件事
--------------------------------------------------------------------
【1】怎么测"解析不可靠输入"
    看「组 3」。我把大模型可能返回的 8 种坏格式全都写成了测试：
    代码块围栏、前置解释、后置总结、对象包数组、单对象、
    字段缺失、百分数置信度、完全不是 JSON。

    这些用例不是我想出来的，是"和 LLM 打交道"这件事本身要求的。
    你的解析器必须扛住全部 —— 而且**一个都不能崩**。

【2】为什么解析要和模型调用分开测
    `parse_response` 不碰模型，所以可以给它喂任意字符串，毫秒级验证。
    如果解析逻辑藏在 `extract` 里面，你就只能通过假模型间接测试它，
    用例会变得又长又绕。

    **能被单独测试的代码，才容易被写对。** 这是设计时就要考虑的，
    不是写完再说。

【3】怎么测"部分成功"
    注意 `test_parse_skips_bad_items_but_keeps_good_ones`。
    一批 5 条里有 2 条格式不对，正确行为是**保留好的 3 条**，
    而不是整批丢掉。

    这类"容错"行为在真实系统里极其重要：
    模型偶尔犯一次错，不该让用户这一轮的记忆全部丢失。

【4】怎么验证"没有多花钱"
    `test_extract_empty_messages_does_not_call_llm` 检查的是：
    对话为空时**一次模型都不许调**。
    每次调用都要花钱，能省的地方必须省。
    这类测试用 FakeLLM 的 call_count 就能精确验证。
"""

import json

import pytest

from memory_assistant.llm import FakeLLM
from memory_assistant.memory.extraction import ExtractedFact, FactExtractor


# ====================================================================
# 测试辅助
# ====================================================================


def make_extractor(llm=None, **kwargs) -> FactExtractor:
    """创建一个抽取器，默认配一个总是返回空数组的假模型。"""
    if llm is None:
        llm = FakeLLM(default_response="[]")
    return FactExtractor(llm, **kwargs)


def item(
    predicate="住在",
    obj="杭州",
    subject="user",
    confidence=1.0,
    evidence="",
    **extra,
) -> dict:
    """构造一个"模型返回的原始字典"。"""
    data = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": confidence,
        "evidence": evidence,
    }
    data.update(extra)
    return data


def as_json(items) -> str:
    """把字典列表转成 JSON 文本（模型返回的样子）。"""
    return json.dumps(items, ensure_ascii=False)


CONVERSATION = [
    {"role": "user", "content": "我叫小明，在杭州做后端开发"},
    {"role": "assistant", "content": "记住了"},
]


# ====================================================================
# 组 1：ExtractedFact 数据模型（规格 S1）
# ====================================================================


def test_fact_text_property():
    fact = ExtractedFact(subject="user", predicate="住在", object="杭州")
    assert fact.text == "user 住在 杭州"


def test_fact_defaults():
    fact = ExtractedFact(predicate="喜欢", object="咖啡")
    assert fact.subject == "user"
    assert fact.confidence == 1.0
    assert fact.evidence == ""


def test_fact_str_shows_confidence():
    fact = ExtractedFact(predicate="住在", object="杭州", confidence=0.85)
    text = str(fact)
    assert "user 住在 杭州" in text
    assert "0.85" in text


def test_fact_requires_predicate_and_object():
    """predicate / object 是必填的 —— 缺了就应该报错，而不是静默变成空字符串。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractedFact(predicate="住在")
    with pytest.raises(ValidationError):
        ExtractedFact(object="杭州")


# ====================================================================
# 组 2：构造与参数校验（规格 S2）
# ====================================================================


def test_rejects_invalid_min_confidence():
    with pytest.raises(ValueError):
        make_extractor(min_confidence=-0.1)
    with pytest.raises(ValueError):
        make_extractor(min_confidence=1.5)


def test_rejects_invalid_max_facts():
    with pytest.raises(ValueError):
        make_extractor(max_facts=0)
    with pytest.raises(ValueError):
        make_extractor(max_facts=-3)


def test_initial_state():
    extractor = make_extractor()
    assert extractor.last_error is None
    assert extractor.last_skipped == 0


# ====================================================================
# 组 3：parse_response 健壮性 ⭐ 本作业的核心
# ====================================================================


def test_parse_clean_json_array():
    extractor = make_extractor()
    raw = as_json([item(predicate="住在", obj="杭州")])

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert facts[0].predicate == "住在"
    assert facts[0].object == "杭州"
    assert facts[0].confidence == pytest.approx(1.0)


def test_parse_strips_markdown_fence_with_language():
    """坏情况 1：模型用 ```json 代码块包起来。"""
    extractor = make_extractor()
    raw = "```json\n" + as_json([item()]) + "\n```"

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert facts[0].object == "杭州"


def test_parse_strips_markdown_fence_without_language():
    extractor = make_extractor()
    raw = "```\n" + as_json([item()]) + "\n```"

    assert len(extractor.parse_response(raw)) == 1


def test_parse_ignores_leading_prose():
    """坏情况 2：前面有一段解释文字。"""
    extractor = make_extractor()
    raw = "好的，我抽取到以下事实：\n" + as_json([item()])

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert facts[0].object == "杭州"


def test_parse_ignores_trailing_prose():
    """坏情况 3：后面还有总结。"""
    extractor = make_extractor()
    raw = as_json([item()]) + "\n以上共 1 条。"

    assert len(extractor.parse_response(raw)) == 1


def test_parse_ignores_prose_on_both_sides():
    extractor = make_extractor()
    raw = "分析结果如下：\n" + as_json([item()]) + "\n希望对你有帮助！"

    assert len(extractor.parse_response(raw)) == 1


def test_parse_object_with_facts_key():
    """坏情况 4：返回对象而不是数组。"""
    extractor = make_extractor()
    raw = as_json({"facts": [item()]})

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert facts[0].object == "杭州"


def test_parse_object_with_arbitrary_list_key():
    """包着数组的字段不一定叫 facts，任何列表字段都该被识别。"""
    extractor = make_extractor()
    raw = as_json({"抽取结果": [item()]})

    assert len(extractor.parse_response(raw)) == 1


def test_parse_single_object_without_array():
    """坏情况 5：只有一条时忘了套数组。"""
    extractor = make_extractor()
    raw = as_json(item())

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert facts[0].predicate == "住在"


def test_parse_empty_array_is_success_not_error():
    """模型说"没抽到东西"是完全正常的，不该被当成失败。"""
    extractor = make_extractor()

    assert extractor.parse_response("[]") == []
    assert extractor.last_error is None


def test_parse_skips_bad_items_but_keeps_good_ones():
    """
    ⭐ 部分成功：一条坏数据不该毁掉整批。

    这批 5 条里有 2 条缺字段，正确的做法是保留好的 3 条。
    """
    extractor = make_extractor()
    raw = as_json(
        [
            item(predicate="住在", obj="杭州"),
            {"subject": "user", "predicate": "职业是"},          # 缺 object
            item(predicate="喜欢", obj="手冲咖啡"),
            {"subject": "user", "object": "豆豆"},                # 缺 predicate
            item(predicate="过敏", obj="花生"),
        ]
    )

    facts = extractor.parse_response(raw)

    assert len(facts) == 3
    assert [f.object for f in facts] == ["杭州", "手冲咖啡", "花生"]
    assert extractor.last_skipped == 2


def test_parse_skips_blank_strings():
    """空字符串和纯空格也算"缺失"，不能让它进库。"""
    extractor = make_extractor()
    raw = as_json(
        [
            item(predicate="住在", obj="杭州"),
            item(predicate="", obj="上海"),
            item(predicate="喜欢", obj="   "),
        ]
    )

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert extractor.last_skipped == 2


def test_parse_strips_whitespace_on_valid_fields():
    extractor = make_extractor()
    raw = as_json([item(subject="  user  ", predicate="  住在 ", obj=" 杭州 ")])

    fact = extractor.parse_response(raw)[0]

    assert fact.subject == "user"
    assert fact.predicate == "住在"
    assert fact.object == "杭州"


def test_parse_defaults_missing_subject_to_user():
    extractor = make_extractor()
    raw = as_json([{"predicate": "住在", "object": "杭州"}])

    assert extractor.parse_response(raw)[0].subject == "user"


def test_parse_defaults_missing_confidence():
    extractor = make_extractor()
    raw = as_json([{"predicate": "住在", "object": "杭州"}])

    assert extractor.parse_response(raw)[0].confidence == pytest.approx(1.0)


def test_parse_non_numeric_confidence_falls_back_to_default():
    """坏情况 7 的变体：confidence 给了个非数字。"""
    extractor = make_extractor()
    raw = as_json([{"predicate": "住在", "object": "杭州", "confidence": "很高"}])

    assert extractor.parse_response(raw)[0].confidence == pytest.approx(1.0)


def test_parse_converts_percentage_confidence():
    """
    坏情况 7：模型经常给 95 而不是 0.95。

    这是实测中非常常见的一种偏差，必须处理 ——
    如果不处理，95 会被 clamp 成 1.0，所有事实都变成"完全确定"。
    """
    extractor = make_extractor()
    raw = as_json(
        [
            {"predicate": "住在", "object": "杭州", "confidence": 95},
            {"predicate": "喜欢", "object": "咖啡", "confidence": 70},
        ]
    )

    facts = extractor.parse_response(raw)

    assert facts[0].confidence == pytest.approx(0.95)
    assert facts[1].confidence == pytest.approx(0.70)


def test_parse_clamps_out_of_range_confidence():
    extractor = make_extractor()
    raw = as_json(
        [
            {"predicate": "住在", "object": "杭州", "confidence": 150},
            {"predicate": "喜欢", "object": "咖啡", "confidence": -0.5},
        ]
    )

    facts = extractor.parse_response(raw)

    assert facts[0].confidence == pytest.approx(1.0)
    assert facts[1].confidence == pytest.approx(0.0)


def test_parse_invalid_json_returns_empty_and_records_error():
    """坏情况 8：根本不是 JSON。"""
    extractor = make_extractor()
    raw = "抱歉，我无法从这段对话中抽取事实。"

    facts = extractor.parse_response(raw)

    assert facts == []
    assert extractor.last_error is not None


def test_parse_empty_string_records_error():
    extractor = make_extractor()

    assert extractor.parse_response("") == []
    assert extractor.last_error is not None


def test_parse_truncated_json_records_error():
    extractor = make_extractor()
    raw = '[{"subject": "user", "predicate": "住在", "object": "杭州"'   # 少了收尾

    facts = extractor.parse_response(raw)

    assert facts == []
    assert extractor.last_error is not None


def test_parse_success_clears_previous_error():
    extractor = make_extractor()

    extractor.parse_response("这不是 JSON")
    assert extractor.last_error is not None

    extractor.parse_response(as_json([item()]))
    assert extractor.last_error is None


def test_parse_resets_last_skipped_each_call():
    extractor = make_extractor()

    extractor.parse_response(as_json([item(), {"predicate": "缺 object"}]))
    assert extractor.last_skipped == 1

    extractor.parse_response(as_json([item()]))
    assert extractor.last_skipped == 0


def test_parse_does_not_raise_on_any_garbage():
    """
    最后一道保险：任何垃圾输入都只能返回列表，**绝对不能抛异常**。

    因为抽取失败不该让整个对话流程崩掉 ——
    大不了这一轮没抽到记忆，用户继续聊天就行。
    """
    extractor = make_extractor()

    garbage_inputs = [
        "",
        "   ",
        "null",
        "123",
        "true",
        "{",
        "}",
        "[",
        "[[[",
        "```",
        "```json",
        "}{",
        '{"facts": "不是列表"}',
        '[{"predicate": null, "object": null}]',
        '[1, 2, 3]',
        '[{"predicate": "住在", "object": "杭州", "confidence": [1, 2]}]',
        "很长的中文说明" * 100,
    ]

    for raw in garbage_inputs:
        result = extractor.parse_response(raw)
        assert isinstance(result, list), f"输入 {raw[:30]!r} 没有返回列表"


def test_parse_skips_non_dict_items():
    """数组里混进了字符串或数字，应该跳过而不是崩。"""
    extractor = make_extractor()
    raw = as_json(["我是一个字符串", 42, None, item()])

    facts = extractor.parse_response(raw)

    assert len(facts) == 1
    assert extractor.last_skipped == 3


# ====================================================================
# 组 4：build_prompt（规格 S3）
# ====================================================================


def prompt_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def test_prompt_is_valid_message_list():
    extractor = make_extractor()
    prompt = extractor.build_prompt(CONVERSATION)

    assert isinstance(prompt, list)
    assert all("role" in m and "content" in m for m in prompt)
    assert all(m["role"] in ("system", "user", "assistant") for m in prompt)


def test_prompt_mentions_extraction_role():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    assert "信息抽取器" in text


def test_prompt_lists_what_to_extract():
    """必须明确告诉模型"抽什么"，否则它会抽一堆没用的东西。"""
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    for keyword in ["姓名", "职业", "偏好", "厌恶", "习惯"]:
        assert keyword in text, f"prompt 的「抽取范围」里缺少：{keyword}"


def test_prompt_lists_what_not_to_extract():
    """也要明确说"不抽什么" —— 不然模型会把"我明天要开会"也记下来。"""
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    for keyword in ["不要抽取", "临时", "寒暄"]:
        assert keyword in text, f"prompt 的「排除范围」里缺少：{keyword}"


def test_prompt_requires_json_array():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    assert "JSON 数组" in text
    assert "[]" in text, "要明确告诉模型：没东西可抽时输出空数组"


def test_prompt_lists_required_fields():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    for field in ["subject", "predicate", "object", "confidence", "evidence"]:
        assert field in text, f"prompt 没有说明字段：{field}"


def test_prompt_contains_the_conversation():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    assert "我叫小明，在杭州做后端开发" in text
    assert "记住了" in text


def test_prompt_formats_conversation_with_roles():
    """对话要按 "role: content" 逐行拼，保留说话人信息。"""
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    assert "user: 我叫小明，在杭州做后端开发" in text
    assert "assistant: 记住了" in text


def test_prompt_omits_known_facts_section_when_not_given():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION))

    assert "已知事实" not in text


def test_prompt_includes_known_facts_when_given():
    """
    把已知事实告诉模型，它就不会重复抽取 —— 这是省钱的实用手段。
    """
    extractor = make_extractor()
    text = prompt_text(
        extractor.build_prompt(CONVERSATION, known_facts=["user 住在 杭州"])
    )

    assert "已知事实" in text
    assert "user 住在 杭州" in text


def test_prompt_omits_known_facts_section_for_empty_list():
    extractor = make_extractor()
    text = prompt_text(extractor.build_prompt(CONVERSATION, known_facts=[]))

    assert "已知事实" not in text


# ====================================================================
# 组 5：extract 完整流程（规格 S5）
# ====================================================================


def test_extract_empty_messages_does_not_call_llm():
    """⭐ 对话为空时一次模型都不许调 —— 每次调用都要花钱。"""
    llm = FakeLLM(default_response=as_json([item()]))
    extractor = make_extractor(llm)

    assert extractor.extract([]) == []
    assert llm.call_count == 0


def test_extract_happy_path():
    llm = FakeLLM(
        default_response=as_json(
            [
                item(predicate="住在", obj="杭州"),
                item(predicate="职业是", obj="后端开发"),
            ]
        )
    )
    extractor = make_extractor(llm)

    facts = extractor.extract(CONVERSATION)

    assert len(facts) == 2
    assert {f.object for f in facts} == {"杭州", "后端开发"}
    assert llm.call_count == 1


def test_extract_filters_low_confidence():
    llm = FakeLLM(
        default_response=as_json(
            [
                {"predicate": "住在", "object": "杭州", "confidence": 0.9},
                {"predicate": "喜欢", "object": "咖啡", "confidence": 0.3},
            ]
        )
    )
    extractor = make_extractor(llm, min_confidence=0.6)

    facts = extractor.extract(CONVERSATION)

    assert [f.object for f in facts] == ["杭州"]


def test_extract_respects_custom_min_confidence():
    llm = FakeLLM(
        default_response=as_json(
            [{"predicate": "喜欢", "object": "咖啡", "confidence": 0.3}]
        )
    )
    extractor = make_extractor(llm, min_confidence=0.2)

    assert len(extractor.extract(CONVERSATION)) == 1


def test_extract_dedups_within_batch():
    """
    批内去重：模型在同一次回答里说了两遍同样的话。

    注意这里刻意用了大小写和空格差异 —— 归一化要处理掉它们。
    """
    llm = FakeLLM(
        default_response=as_json(
            [
                item(subject="user", predicate="住在", obj="杭州", confidence=0.8),
                item(subject="User", predicate=" 住在 ", obj=" 杭州 ", confidence=0.9),
            ]
        )
    )
    extractor = make_extractor(llm)

    facts = extractor.extract(CONVERSATION)

    assert len(facts) == 1
    assert facts[0].confidence == pytest.approx(0.9), "去重后应该保留置信度更高的那条"


def test_extract_keeps_distinct_facts():
    """不同的谓词是不同的信息，不能合并。"""
    llm = FakeLLM(
        default_response=as_json(
            [
                item(predicate="住在", obj="杭州"),
                item(predicate="工作在", obj="杭州"),
            ]
        )
    )
    extractor = make_extractor(llm)

    assert len(extractor.extract(CONVERSATION)) == 2


def test_extract_sorts_by_confidence_desc():
    llm = FakeLLM(
        default_response=as_json(
            [
                {"predicate": "住在", "object": "杭州", "confidence": 0.7},
                {"predicate": "职业是", "object": "后端", "confidence": 0.99},
                {"predicate": "喜欢", "object": "咖啡", "confidence": 0.85},
            ]
        )
    )
    extractor = make_extractor(llm)

    confidences = [f.confidence for f in extractor.extract(CONVERSATION)]

    assert confidences == sorted(confidences, reverse=True)


def test_extract_respects_max_facts():
    llm = FakeLLM(
        default_response=as_json(
            [
                {"predicate": f"谓词{i}", "object": f"值{i}", "confidence": 0.9}
                for i in range(10)
            ]
        )
    )
    extractor = make_extractor(llm, max_facts=3)

    facts = extractor.extract(CONVERSATION)

    assert len(facts) == 3


def test_extract_max_facts_keeps_most_confident():
    llm = FakeLLM(
        default_response=as_json(
            [
                {"predicate": "低", "object": "x", "confidence": 0.61},
                {"predicate": "高", "object": "x", "confidence": 0.99},
                {"predicate": "中", "object": "x", "confidence": 0.80},
            ]
        )
    )
    extractor = make_extractor(llm, max_facts=1)

    assert extractor.extract(CONVERSATION)[0].predicate == "高"


def test_extract_handles_llm_failure():
    """模型挂了：返回空列表、记录错误、**不抛异常**。"""

    class BrokenLLM:
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, **kwargs):
            self.call_count += 1
            raise RuntimeError("模拟网络故障")

    extractor = make_extractor(BrokenLLM())

    assert extractor.extract(CONVERSATION) == []
    assert isinstance(extractor.last_error, RuntimeError)


def test_extract_handles_bad_json_from_llm():
    llm = FakeLLM(default_response="我不太确定，可能是杭州吧")
    extractor = make_extractor(llm)

    assert extractor.extract(CONVERSATION) == []
    assert extractor.last_error is not None


def test_extract_passes_known_facts_into_prompt():
    llm = FakeLLM(default_response="[]")
    extractor = make_extractor(llm)

    extractor.extract(CONVERSATION, known_facts=["user 住在 杭州"])

    sent = prompt_text(llm.calls[-1])
    assert "已知事实" in sent
    assert "user 住在 杭州" in sent


def test_extract_returns_extracted_fact_objects():
    """返回的必须是 ExtractedFact 对象，不是原始字典 —— 这样才有类型保障。"""
    llm = FakeLLM(default_response=as_json([item()]))
    extractor = make_extractor(llm)

    facts = extractor.extract(CONVERSATION)

    assert all(isinstance(f, ExtractedFact) for f in facts)
    # 而且能直接拿到 text，喂给 LongTermMemory
    assert facts[0].text == "user 住在 杭州"


# ====================================================================
# 组 6：repr（规格 S6）
# ====================================================================


def test_repr_is_informative():
    extractor = make_extractor(min_confidence=0.7, max_facts=5)
    text = repr(extractor)

    assert "FactExtractor" in text
    assert "0.7" in text
    assert "5" in text
