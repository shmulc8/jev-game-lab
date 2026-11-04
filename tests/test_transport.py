import unittest
from unittest.mock import patch
import jevhttp


class TransportTests(unittest.TestCase):
    def test_rejects_non_official_destinations_before_sending_credentials(self):
        with patch.object(jevhttp, "_connection") as connect:
            for url in ["http://api.typesafe.ai/v1/systemone", "https://example.com/", "https://api.typesafe.ai.evil.test/", "https://user@api.typesafe.ai/", "https://api.typesafe.ai:444/"]:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    jevhttp.post_json(url, {}, "test-placeholder")
            connect.assert_not_called()

    def test_preserves_official_query_and_applies_https_transport(self):
        with patch.object(jevhttp, "_connection") as connect:
            response = connect.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = b"{}"
            response.getheader.return_value = "keep-alive"
            status, body = jevhttp.post_json("https://api.typesafe.ai/v1/systemone?mode=test", {}, "test-placeholder", timeout=3)
            self.assertEqual((status, body), (200, b"{}"))
            self.assertEqual(connect.call_args.args, ("api.typesafe.ai", 3))
            args, kwargs = connect.return_value.request.call_args
            self.assertEqual(args, ("POST", "/v1/systemone?mode=test"))
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-placeholder")


if __name__ == "__main__":
    unittest.main()
