import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p25.data import load_with_download_retry


class DownloadRetryTests(unittest.TestCase):
    @patch('p25.data.time.sleep')
    def test_http_500_retries_then_succeeds(self, sleep):
        error = HTTPError('https://example.invalid/data', 500, 'Internal Server Error', {}, None)
        factory = Mock(side_effect=[error, error, 'loaded'])
        self.assertEqual(load_with_download_retry(factory, True), 'loaded')
        self.assertEqual(factory.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    @patch('p25.data.time.sleep')
    def test_permanent_http_error_is_not_retried(self, sleep):
        error = HTTPError('https://example.invalid/data', 404, 'Not Found', {}, None)
        factory = Mock(side_effect=error)
        with self.assertRaises(HTTPError):
            load_with_download_retry(factory, True)
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_called()

    @patch('p25.data.time.sleep')
    def test_network_retry_is_bounded_and_requires_download(self, sleep):
        factory = Mock(side_effect=URLError('connection failed'))
        with self.assertRaises(URLError):
            load_with_download_retry(factory, True)
        self.assertEqual(factory.call_count, 3)
        factory.reset_mock()
        sleep.reset_mock()
        with self.assertRaises(URLError):
            load_with_download_retry(factory, False)
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
