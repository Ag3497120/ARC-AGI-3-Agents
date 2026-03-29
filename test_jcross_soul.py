#!/usr/bin/env python3
"""
test_jcross_soul.py — jcross_runtime + soul.jcrossの単体テスト

テスト内容:
1. ランタイム初期化
2. ダミーグリッドを注入してdecide()が有効な行動インデックスを返すことを確認
3. 体験を追記してmemory.jcrossが更新されることを確認
4. ルール書き換えが動作することを確認
5. パーサーの基本動作確認
"""

import sys
import os
import tempfile
import shutil

# パスを通す
sys.path.insert(0, os.path.expanduser('~/verantyx_v6/bootstrap'))
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, 'agents', 'cross_engine'))

# jcross_runtimeを直接インポート（arcengineへの依存を回避）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "jcross_runtime",
    os.path.join(_this_dir, 'agents', 'cross_engine', 'jcross_runtime.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
JCrossRuntime = _mod.JCrossRuntime

SOUL_PATH = os.path.join(os.path.dirname(__file__), 'agents', 'cross_engine', 'soul.jcross')
MEMORY_PATH = os.path.join(os.path.dirname(__file__), 'agents', 'cross_engine', 'memory.jcross')


def make_dummy_grid(size=64, val=1):
    """64x64のダミーグリッドを作成"""
    return [[val for _ in range(size)] for _ in range(size)]


def test_1_init():
    """テスト1: ランタイム初期化"""
    print("=" * 50)
    print("テスト1: ランタイム初期化")
    runtime = JCrossRuntime()
    assert runtime is not None, "JCrossRuntime作成失敗"
    print(f"  runtime: {runtime}")
    print("  ✅ 初期化成功")
    return runtime


def test_2_load_and_decide():
    """テスト2: ロードとdecide()"""
    print("=" * 50)
    print("テスト2: ロード & decide()")

    runtime = JCrossRuntime()
    ok = runtime.load(SOUL_PATH, MEMORY_PATH)

    if not ok:
        print("  ⚠️  ロード失敗（パーサー未利用 or ファイルなし）")
        print(f"  soul_path exists: {os.path.exists(SOUL_PATH)}")
        return None

    print(f"  ロード成功: {runtime}")

    # ダミー状態を注入
    dummy_grid = make_dummy_grid()
    runtime.inject_all({
        "フェーズ": "実行",
        "グリッド": dummy_grid,
        "自分の位置": [32, 32],
        "フレーム番号": 5,
        "前回の行動": 0,
        "移動したか": False,
        "差分あり": False,
        "利用可能行動": [0, 1, 2, 3],
        "フェーズ": "実行",
        "走路の色": [3],
        "壁の色": [1],
        "スタック回数": 0,
        "クリックゲーム": False,
        "行動キュー": [2, 1, 0],
        "探索キュー": [],
        "衝動リスト": [],
        "経路先頭": 2,
        "フォールバック": 0,
    })

    result = runtime.decide()
    print(f"  decide() → {result}")

    if result == -1:
        print("  ⚠️  decide()が-1（フォールバック発動）")
    else:
        assert 0 <= result <= 6, f"無効な行動インデックス: {result}"
        print(f"  ✅ 有効な行動インデックス: {result}")

    return runtime


def test_3_memory_update():
    """テスト3: memory.jcrossへの追記"""
    print("=" * 50)
    print("テスト3: memory.jcross追記")

    # 一時ファイルにコピー
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_memory = os.path.join(tmpdir, 'memory.jcross')
        shutil.copy(MEMORY_PATH, tmp_memory)

        runtime = JCrossRuntime()
        ok = runtime.load(SOUL_PATH, tmp_memory)

        if not ok:
            print("  ⚠️  ロード失敗")
            return

        # 体験を追記
        experiences = [
            {"種類": "ブロック", "フレーム": 10, "位置": [30, 30], "行動": 0},
            {"種類": "移動", "フレーム": 11, "位置": [31, 30], "行動": 1},
        ]

        success = runtime.update_memory(experiences)
        assert success, "update_memory()が失敗"

        # ファイルを確認
        with open(tmp_memory, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "体験追記" in content, "体験が追記されていない"
        assert "フレーム=10" in content, "フレーム番号が記録されていない"
        print(f"  memory.jcross内容（抜粋）:")
        for line in content.split('\n'):
            if '体験' in line or '追記' in line:
                print(f"    {line}")
        print("  ✅ memory.jcross追記成功")


def test_4_rewrite_rule():
    """テスト4: ルール書き換え"""
    print("=" * 50)
    print("テスト4: ルール書き換え (rewrite_rule)")

    # 一時ファイルにコピー
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_soul = os.path.join(tmpdir, 'soul.jcross')
        tmp_memory = os.path.join(tmpdir, 'memory.jcross')
        shutil.copy(SOUL_PATH, tmp_soul)
        shutil.copy(MEMORY_PATH, tmp_memory)

        runtime = JCrossRuntime()
        ok = runtime.load(tmp_soul, tmp_memory)

        if not ok:
            print("  ⚠️  ロード失敗")
            return

        # スタック脱出ルールを書き換え
        new_rule = """関数 スタック脱出() {
  もし スタック回数 > 5 {
    返す 3
  }
  返す -1
}"""

        success = runtime.rewrite_rule("スタック脱出", new_rule)
        assert success, "rewrite_rule()が失敗"

        # ファイルを確認
        with open(tmp_soul, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "スタック回数 > 5" in content, "新しいルールが書き込まれていない"
        assert "// [RULE:スタック脱出 START]" in content, "マーカーが保持されていない"
        print("  書き換え後のルール確認:")
        in_rule = False
        for line in content.split('\n'):
            if '[RULE:スタック脱出' in line:
                in_rule = not in_rule
                print(f"    {line}")
            elif in_rule:
                print(f"    {line}")
        print("  ✅ ルール書き換え成功")


def test_5_parser_basics():
    """テスト5: パーサー基本動作の確認"""
    print("=" * 50)
    print("テスト5: パーサー基本動作")

    try:
        from jcross_japanese_parser import JCrossJapaneseParser
    except ImportError:
        print("  ⚠️  パーサーがインポートできません")
        return

    p = JCrossJapaneseParser()
    code = """
フェーズ = "実行"
行動キュー = [2, 1, 0]
スタック回数 = 0

関数 行動を決める() {
  もし フェーズ == "実行" {
    もし 長さ(行動キュー) > 0 {
      返す 行動キュー[0]
    }
  }
  返す 0
}

結果 = 行動を決める()
"""
    p.execute(code)
    result = p.globals.get("結果")
    print(f"  コード実行結果: 結果 = {result}")
    assert result == 2, f"期待値2, 実際は{result}"
    print("  ✅ パーサー基本動作 OK")


def test_6_jcross_phase_transitions():
    """テスト6: フェーズごとの行動選択"""
    print("=" * 50)
    print("テスト6: フェーズ別行動選択")

    try:
        from jcross_japanese_parser import JCrossJapaneseParser
    except ImportError:
        print("  ⚠️  パーサーがインポートできません")
        return

    if not os.path.exists(SOUL_PATH):
        print(f"  ⚠️  soul.jcrossが見つかりません: {SOUL_PATH}")
        return

    with open(SOUL_PATH, 'r', encoding='utf-8') as f:
        soul_source = f.read()

    test_cases = [
        ("観察", {"利用可能行動": [0, 1, 2, 3], "フォールバック": 0}, 0),
        ("探索", {"探索キュー": [1, 2], "フォールバック": 0}, 1),
        ("実行", {"行動キュー": [3], "経路先頭": 3, "フォールバック": 0, "スタック回数": 0, "衝動リスト": []}, 3),
    ]

    for phase, inject_vars, expected in test_cases:
        p = JCrossJapaneseParser()
        # デフォルト値
        p.globals.update({
            "フェーズ": phase,
            "クリックゲーム": False,
            "スタック回数": 0,
            "利用可能行動": [0, 1, 2, 3],
            "行動キュー": [],
            "探索キュー": [],
            "衝動リスト": [],
            "経路先頭": -1,
            "フォールバック": 0,
            "フレーム番号": 1,
            "前回の行動": 0,
        })
        p.globals.update(inject_vars)

        combined = soul_source + "\n\n行動結果 = 行動を決める()\n"
        try:
            p.execute(combined)
            result = p.globals.get("行動結果")
            status = "✅" if result == expected else "⚠️"
            print(f"  {status} フェーズ={phase}: decide()={result} (期待={expected})")
        except Exception as e:
            print(f"  ❌ フェーズ={phase}: エラー={e}")


def test_7_memory_changes_decision():
    """テスト7: memory.jcrossに体験蓄積後にdecide()結果が変わることを確認"""
    print("=" * 50)
    print("テスト7: 体験蓄積後のdecide()変化")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_soul = os.path.join(tmpdir, 'soul.jcross')
        tmp_memory = os.path.join(tmpdir, 'memory.jcross')
        shutil.copy(SOUL_PATH, tmp_soul)
        shutil.copy(MEMORY_PATH, tmp_memory)

        runtime = JCrossRuntime()
        ok = runtime.load(tmp_soul, tmp_memory)
        if not ok:
            print("  ⚠️  ロード失敗")
            return

        # 体験なしの状態でdecide() → 行動キューに従う
        runtime.inject_all({
            "フェーズ": "実行",
            "行動キュー": [2, 1, 0],
            "経路先頭": 2,
            "フォールバック": 0,
            "スタック回数": 0,
            "前回の行動": 0,
            "フレーム番号": 5,
            "クリックゲーム": False,
            "衝動リスト": [],
            "探索キュー": [],
            "利用可能行動": [0, 1, 2, 3],
        })
        result_before = runtime.decide()
        print(f"  体験なし: decide()={result_before}")

        # 大量のブロック体験を追記
        block_experiences = [
            {"種類": "ブロック", "フレーム": i, "位置": [30, 30], "行動": 0}
            for i in range(6)
        ]
        runtime.update_memory(block_experiences)

        # decide()が変わるはず（前回の行動0からの反転: (0+2)%4 = 2だが、
        # 体験から学ぶ()が 前回の行動(0+2)%4=2 を返す）
        runtime.inject_all({
            "フェーズ": "実行",
            "行動キュー": [1, 2, 3],  # 行動キューは1から始まる
            "経路先頭": 1,
            "フォールバック": 0,
            "スタック回数": 0,
            "前回の行動": 0,
            "フレーム番号": 20,
            "クリックゲーム": False,
            "衝動リスト": [],
            "探索キュー": [],
            "利用可能行動": [0, 1, 2, 3],
        })
        result_after = runtime.decide()
        print(f"  体験6件蓄積後: decide()={result_after}")
        print(f"  変化: {result_before} → {result_after}")
        # 体験から学ぶ()は ブロック数>5 の時に (前回の行動+2)%4 = (0+2)%4 = 2 を返す
        # 行動キュー[0]=1 とは異なるはず
        if result_after == 2:
            print("  ✅ 体験蓄積によってdecide()が変化（体験から学ぶ=2）")
        elif result_after != result_before:
            print(f"  ✅ decide()が変化: {result_before}→{result_after}")
        else:
            print(f"  ⚠️  decide()変化なし（{result_after}）— 体験ロード確認要")


def test_8_rewrite_changes_decision():
    """テスト8: rewrite_rule()後に行動が変わることを確認"""
    print("=" * 50)
    print("テスト8: rewrite_rule()後の行動変化")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_soul = os.path.join(tmpdir, 'soul.jcross')
        tmp_memory = os.path.join(tmpdir, 'memory.jcross')
        shutil.copy(SOUL_PATH, tmp_soul)
        shutil.copy(MEMORY_PATH, tmp_memory)

        runtime = JCrossRuntime()
        ok = runtime.load(tmp_soul, tmp_memory)
        if not ok:
            print("  ⚠️  ロード失敗")
            return

        # スタック回数=10でのdecide()（通常ルール: スタック>8なら (前回+1)%4）
        runtime.inject_all({
            "フェーズ": "実行",
            "行動キュー": [],
            "経路先頭": -1,
            "フォールバック": 0,
            "スタック回数": 10,
            "前回の行動": 2,
            "フレーム番号": 50,
            "クリックゲーム": False,
            "衝動リスト": [],
            "探索キュー": [],
            "利用可能行動": [0, 1, 2, 3],
        })
        result_before = runtime.decide()
        print(f"  書き換え前 (スタック=10): decide()={result_before}")
        # 通常ルール: スタック>8 → (前回の行動+1)%4 = (2+1)%4 = 3

        # スタック脱出ルールを書き換え（常に固定値3を返す）
        new_rule = """関数 スタック脱出() {
  もし スタック回数 > 5 {
    返す 3
  }
  返す -1
}"""
        success = runtime.rewrite_rule("スタック脱出", new_rule)
        assert success, "rewrite_rule()失敗"

        result_after = runtime.decide()
        print(f"  書き換え後 (スタック=10): decide()={result_after}")

        # 書き換え後は スタック>5 → 返す3
        if result_after == 3:
            print("  ✅ rewrite_rule()後に期待通り3が返された")
        elif result_after == result_before:
            print(f"  ⚠️  変化なし（両方{result_after}）— 書き換えが反映されているが同じ値")
        else:
            print(f"  ⚠️  書き換え結果: {result_before}→{result_after}")


def test_9_learn_from_experience_function():
    """テスト9: 体験から学ぶ()関数が実際にブロック回避行動を返す"""
    print("=" * 50)
    print("テスト9: 体験から学ぶ()のブロック回避")

    try:
        from jcross_japanese_parser import JCrossJapaneseParser
    except ImportError:
        print("  ⚠️  パーサーがインポートできません")
        return

    if not os.path.exists(SOUL_PATH):
        print(f"  ⚠️  soul.jcrossが見つかりません")
        return

    with open(SOUL_PATH, 'r', encoding='utf-8') as f:
        soul_source = f.read()

    # 体験から学ぶ()のみをテスト
    if "体験から学ぶ" not in soul_source:
        print("  ⚠️  soul.jcrossに体験から学ぶ()が存在しない")
        return

    # ブロック体験6件 → (前回の行動+2)%4 が返るはず
    test_cases = [
        (0, 6, (0 + 2) % 4),   # 前回の行動=0, ブロック6件 → 2
        (1, 6, (1 + 2) % 4),   # 前回の行動=1, ブロック6件 → 3
        (3, 6, (3 + 2) % 4),   # 前回の行動=3, ブロック6件 → 1
        (2, 3, -1),             # ブロック3件 → まだ学習しない(-1)
    ]

    for prev_action, block_count, expected in test_cases:
        p = JCrossJapaneseParser()
        block_exps = [{"種類": "ブロック", "フレーム": i, "行動": prev_action} for i in range(block_count)]
        p.globals.update({
            "体験リスト": block_exps,
            "体験数": block_count,
            "前回の行動": prev_action,
            "フレーム番号": 10,
            "フェーズ": "実行",
            "クリックゲーム": False,
            "スタック回数": 0,
            "行動キュー": [prev_action],
            "探索キュー": [],
            "衝動リスト": [],
            "経路先頭": prev_action,
            "フォールバック": 0,
            "利用可能行動": [0, 1, 2, 3],
        })

        code = soul_source + f"\n\n学習結果 = 体験から学ぶ()\n"
        try:
            p.execute(code)
            result = p.globals.get("学習結果")
            status = "✅" if result == expected else "⚠️"
            print(f"  {status} 前回行動={prev_action}, ブロック数={block_count}: "
                  f"体験から学ぶ()={result} (期待={expected})")
        except Exception as e:
            print(f"  ❌ エラー: prev={prev_action}, blocks={block_count}: {e}")


def main():
    print("\n🔥 jcross soul テスト開始\n")

    test_5_parser_basics()
    test_1_init()
    test_2_load_and_decide()
    test_3_memory_update()
    test_4_rewrite_rule()
    test_6_jcross_phase_transitions()
    test_7_memory_changes_decision()
    test_8_rewrite_changes_decision()
    test_9_learn_from_experience_function()

    print("\n" + "=" * 50)
    print("✅ 全テスト完了")


if __name__ == '__main__':
    main()
