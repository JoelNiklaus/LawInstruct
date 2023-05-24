"""Tests for multiple choice styles."""

from absl.testing import absltest

import multiple_choice


class MultipleChoiceTest(absltest.TestCase):
    """Tests for multiple choice styles."""

    def test_one_style(self):
        # Given a multiple choice style...
        style = multiple_choice.MultipleChoiceStyle(['a>', 'b>', 'c>', 'd>'])
        # ...and some options...
        options = ['foo', 'bar', 'baz']
        # ...when you get the markers for those options...
        markers = style.markers_for_options(options)
        # ...then you should get the right markers.
        expected_markers = ['a>', 'b>', 'c>']
        self.assertListEqual(markers, expected_markers)

    def test_sampling(self):
        # Given a list of options...
        options = ['foo', 'bar', 'baz']
        # ...when you sample markers for those options...
        markers = multiple_choice.sample_markers_for_options(options)
        # ...then you should get the right number of markers.
        self.assertLen(markers, len(options))


if __name__ == '__main__':
    absltest.main()