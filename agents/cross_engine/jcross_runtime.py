"""
jcross_runtime.py — .jcrossファイルをARC-AGI-3エージェントから実行するランタイム

Pythonからjcross世界にグリッド・プレイヤー位置・体験を注入し、
jcross世界から行動決定を受け取る。

核心: Pythonはセンサー（目と手）。jcrossは脳。
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# パーサーのパスを追加
sys.path.insert(0, os.path.expanduser('~/verantyx_v6/bootstrap'))

try:
    from jcross_japanese_parser import JCrossJapaneseParser
    _PARSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"jcross_japanese_parser import failed: {e}")
    _PARSER_AVAILABLE = False


class JCrossRuntime:
    """
    jcrossランタイム — ARC-AGI-3エージェントの脳

    使い方:
        runtime = JCrossRuntime()
        runtime.load(soul_path, memory_path)
        runtime.inject("フェーズ", "観察")
        runtime.inject("行動キュー", [1, 2, 3])
        action_idx = runtime.decide()
    """

    def __init__(self):
        self._soul_path: Optional[str] = None
        self._memory_path: Optional[str] = None
        self._soul_source: str = ""
        self._memory_source: str = ""
        self._injected: Dict[str, Any] = {}
        self._parser: Optional[Any] = None
        self._available = _PARSER_AVAILABLE

        if not self._available:
            logger.warning("JCrossRuntime: パーサーが利用できません。Pythonフォールバックを使用します。")

    def load(self, soul_path: str, memory_path: Optional[str] = None) -> bool:
        """
        .jcrossファイルを読み込む

        Args:
            soul_path: soul.jcrossのパス
            memory_path: memory.jcrossのパス（省略可）

        Returns:
            True: 成功 / False: 失敗
        """
        if not self._available:
            return False

        try:
            soul_path = os.path.expanduser(soul_path)
            self._soul_path = soul_path

            with open(soul_path, 'r', encoding='utf-8') as f:
                self._soul_source = f.read()

            if memory_path:
                memory_path = os.path.expanduser(memory_path)
                self._memory_path = memory_path
                if os.path.exists(memory_path):
                    with open(memory_path, 'r', encoding='utf-8') as f:
                        self._memory_source = f.read()
                else:
                    self._memory_source = ""

            logger.info(f"JCrossRuntime: loaded soul={soul_path}")
            return True

        except Exception as e:
            logger.error(f"JCrossRuntime.load error: {e}")
            return False

    def inject(self, name: str, value: Any) -> None:
        """
        Python値をjcross空間に注入する

        Args:
            name: 変数名（日本語可）
            value: 値（グリッド、位置、体験リスト等）
        """
        self._injected[name] = value

    def inject_all(self, variables: Dict[str, Any]) -> None:
        """複数の変数を一括注入"""
        for name, value in variables.items():
            self._injected[name] = value

    def decide(self) -> int:
        """
        jcrossのsoulを実行して行動インデックスを返す

        Returns:
            行動インデックス (0-5)。失敗時は -1（Pythonフォールバックを示す）
        """
        if not self._available or not self._soul_source:
            return -1

        try:
            # 毎回フレッシュなパーサーを作成（状態汚染を避けるため）
            parser = JCrossJapaneseParser()

            # 注入された変数をパーサーのglobalsに設定
            for name, value in self._injected.items():
                parser.globals[name] = value

            # memory.jcrossを先に実行（体験を読み込む）
            if self._memory_source:
                try:
                    mem_parser = JCrossJapaneseParser()
                    for name, value in self._injected.items():
                        mem_parser.globals[name] = value
                    mem_parser.execute(self._memory_source)
                    # memory.jcrossで定義された変数をsoulに持ち込む
                    for key, val in mem_parser.globals.items():
                        if key not in parser.globals:
                            parser.globals[key] = val
                except Exception as e:
                    logger.debug(f"memory.jcross exec skip: {e}")

            # soul.jcrossを実行
            combined_source = self._soul_source + "\n\n// ===== 行動決定 =====\n行動結果 = 行動を決める()\n"
            parser.execute(combined_source)

            result = parser.globals.get("行動結果")

            if result is None:
                result = parser.return_value

            if isinstance(result, int):
                return result
            elif isinstance(result, float):
                return int(result)
            else:
                logger.debug(f"decide: unexpected result type={type(result)} val={result}")
                return -1

        except Exception as e:
            logger.warning(f"JCrossRuntime.decide error: {e}")
            return -1

    def get_state(self, var_name: str, default: Any = None) -> Any:
        """
        jcross実行後の変数値を取得する
        decide()の後に呼び出すこと

        Args:
            var_name: 変数名
            default: デフォルト値

        Returns:
            変数の値
        """
        # 最後のパーサー状態を保持するには、decide()を改修が必要
        # 現在は注入済み変数のみ返す
        return self._injected.get(var_name, default)

    def update_memory(self, experiences: List[Dict]) -> bool:
        """
        体験をmemory.jcrossに追記して保存（動的書き換え）

        Args:
            experiences: 体験リスト [{"種類": str, "フレーム": int, "位置": [...], ...}]

        Returns:
            True: 成功
        """
        if not self._memory_path:
            return False

        try:
            lines = []
            for exp in experiences:
                # jcross辞書リテラルとして追記
                parts = []
                for key, val in exp.items():
                    if isinstance(val, str):
                        parts.append(f'"{key}": "{val}"')
                    elif isinstance(val, (list, tuple)):
                        inner = ", ".join(str(v) for v in val)
                        parts.append(f'"{key}": [{inner}]')
                    else:
                        parts.append(f'"{key}": {val}')
                dict_str = "{" + ", ".join(parts) + "}"
                lines.append(f"// 体験追記: フレーム={exp.get('フレーム', '?')}")
                lines.append(f"体験リスト に 追加({dict_str})")
                lines.append(f"体験数 = 体験数 + 1")

            if lines:
                with open(self._memory_path, 'a', encoding='utf-8') as f:
                    f.write("\n")
                    f.write("\n".join(lines))
                    f.write("\n")

                # メモリソースを再読み込み
                with open(self._memory_path, 'r', encoding='utf-8') as f:
                    self._memory_source = f.read()

            return True

        except Exception as e:
            logger.error(f"JCrossRuntime.update_memory error: {e}")
            return False

    def rewrite_rule(self, rule_id: str, new_rule_jcross: str) -> bool:
        """
        soul.jcrossのルール部分を動的に書き換える

        マーカー形式:
            // [RULE:rule_id START]
            関数 ... { ... }
            // [RULE:rule_id END]

        Args:
            rule_id: ルールID（マーカーに使う識別子）
            new_rule_jcross: 新しいjcrossコード

        Returns:
            True: 成功
        """
        if not self._soul_path:
            return False

        try:
            with open(self._soul_path, 'r', encoding='utf-8') as f:
                source = f.read()

            start_marker = f"// [RULE:{rule_id} START]"
            end_marker = f"// [RULE:{rule_id} END]"

            start_idx = source.find(start_marker)
            end_idx = source.find(end_marker)

            if start_idx == -1 or end_idx == -1:
                # マーカーがない場合は末尾に追加
                new_block = f"\n{start_marker}\n{new_rule_jcross}\n{end_marker}\n"
                source = source + new_block
                logger.info(f"rewrite_rule: added new rule block '{rule_id}'")
            else:
                # 既存のブロックを置換
                before = source[:start_idx]
                after = source[end_idx + len(end_marker):]
                new_block = f"{start_marker}\n{new_rule_jcross}\n{end_marker}"
                source = before + new_block + after
                logger.info(f"rewrite_rule: replaced rule block '{rule_id}'")

            with open(self._soul_path, 'w', encoding='utf-8') as f:
                f.write(source)

            # ソースを再読み込み
            self._soul_source = source
            return True

        except Exception as e:
            logger.error(f"JCrossRuntime.rewrite_rule error: {e}")
            return False

    def reload(self) -> bool:
        """soul.jcrossとmemory.jcrossを再読み込みする"""
        if self._soul_path:
            return self.load(self._soul_path, self._memory_path)
        return False

    @property
    def is_available(self) -> bool:
        """パーサーが利用可能かどうか"""
        return self._available and bool(self._soul_source)

    def __repr__(self):
        return (f"JCrossRuntime(available={self._available}, "
                f"soul={'loaded' if self._soul_source else 'none'}, "
                f"memory={'loaded' if self._memory_source else 'none'}, "
                f"injected={list(self._injected.keys())})")
