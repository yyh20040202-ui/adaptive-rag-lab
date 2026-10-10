from adaptive_rag.answerability import has_topical_overlap, missing_or_echo_answer, missing_query_anchor


def test_generic_words_do_not_make_missing_topics_answerable():
    assert not has_topical_overlap("What is the quaternion insurance rate?", ["The hotel allowance is CNY 500."])
    assert not has_topical_overlap("公司预计营业收入是多少？", ["年假至少提前五个工作日申请；须在考勤系统留痕。"])
    assert has_topical_overlap("客户数据可以上传公共AI服务吗？", ["客户数据不得上传公共AI服务。"])


def test_missing_fact_guard_keeps_valid_negative_instructions():
    assert missing_or_echo_answer("Annual budget?", "The annual budget is not specified.") == "answer_reports_missing_fact"
    assert missing_or_echo_answer("What is the budget?", "What is the budget?") == "question_echo"
    assert missing_or_echo_answer("Can I upload customer data?", "Do not upload customer data to public AI services.") is None


def test_cosmetic_article_changes_are_echoes_without_erasing_negation():
    assert missing_or_echo_answer("The TET protein loss causes cancer.", "TET protein loss causes cancer.") == "question_echo"
    assert missing_or_echo_answer("The TET protein loss causes cancer.", "Supported: TET protein loss causes cancer.") == "question_echo"
    assert missing_or_echo_answer("The TET protein loss causes cancer.", "Supported in mice: the knockout study found myeloid transformation.") is None
    assert missing_or_echo_answer("The protein does not cause cancer.", "The protein does cause cancer.") is None


def test_formal_query_anchors_require_presence_in_supplied_text():
    assert missing_query_anchor("0-dimensional biomaterials have inductive properties.", ["Biomaterials activate immune cells."]) == "0-dimensional"
    assert missing_query_anchor("0-dimensional biomaterials have inductive properties.", ["0-dimensional biomaterials were investigated."]) is None
    assert missing_query_anchor("TET protein loss alters cells.", ["Tet2 loss initiates myeloid transformation."]) is None
    assert missing_query_anchor("polI loss changes DNA repair.", ["DNA polymerase eta controls UV repair."]) == "polI"
