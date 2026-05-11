# Copyright 2026-present, thekozugroup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest

from hadron.models.image_generation.openai import OpenAIImageGeneration


@patch("openai.OpenAI")
@patch("openai.AsyncOpenAI")
class TestOpenAIImageGeneration:
    model_id: str = "dall-e-3"

    @patch("requests.Session")
    def test_openai_image_generation(
        self,
        mock_session: MagicMock,
        _async_openai_mock: MagicMock,
        _openai_mock: MagicMock,
    ):
        igm = OpenAIImageGeneration(
            model="dall-e-3",
            api_key="api.key",
            generation_kwargs={
                "size": "1024x1024",
                "quality": "standard",
                "style": "natural",
            },
        )

        assert isinstance(igm, OpenAIImageGeneration)
        assert igm.model_name == self.model_id

        igm.load()
        assert igm._session is not None
        mock_session.assert_called_once()

        igm.unload()
        assert igm._session is None
        mock_session.return_value.close.assert_called_once()

    @pytest.mark.parametrize("response_format", ["url", "b64_json"])
    @pytest.mark.asyncio
    async def test_agenerate(
        self,
        async_openai_mock: MagicMock,
        _openai_mock: MagicMock,
        response_format: str,
    ) -> None:
        igm = OpenAIImageGeneration(model=self.model_id, api_key="api.key")  # type: ignore
        igm._aclient = async_openai_mock
        igm._session = MagicMock()

        with patch.object(igm._session, "get") as mock_get:
            # Mock the download of the image
            mock_get.return_value = Mock(content=b"iVBORw0KGgoAAAANSUhEUgA...")
            if response_format == "url":
                mocked_response = Mock(b64_json=None, url="https://example.com")
            else:
                mocked_response = Mock(b64_json="iVBORw0KGgoAAAANSUhEUgA...", url=None)

            mocked_generation = Mock(data=[mocked_response])
            igm._aclient.images.generate = AsyncMock(return_value=mocked_generation)

            await igm.agenerate(
                input="a white siamese cat", response_format=response_format
            )

    @pytest.mark.parametrize("response_format", ["url", "b64_json"])
    @pytest.mark.asyncio
    async def test_generate(
        self,
        async_openai_mock: MagicMock,
        _openai_mock: MagicMock,
        response_format: str,
    ) -> None:
        igm = OpenAIImageGeneration(model=self.model_id, api_key="api.key")  # type: ignore
        igm._aclient = async_openai_mock
        igm._session = MagicMock()

        with patch.object(igm._session, "get") as mock_get:
            # Mock the download of the image
            mock_get.return_value = Mock(content=b"iVBORw0KGgoAAAANSUhEUgA...")

            if response_format == "url":
                mocked_response = Mock(b64_json=None, url="https://example.com")
            else:
                mocked_response = Mock(b64_json="iVBORw0KGgoAAAANSUhEUgA...", url=None)

            mocked_generation = Mock(data=[mocked_response])
            igm._aclient.images.generate = AsyncMock(return_value=mocked_generation)

            nest_asyncio.apply()

            igm.generate(
                inputs=["a white siamese cat"], response_format=response_format
            )

        with pytest.raises(ValueError):
            igm.generate(
                inputs=[
                    "a white siamese cat",
                ],
                response_format="unkown_format",
            )
