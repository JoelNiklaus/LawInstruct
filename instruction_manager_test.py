from absl.testing import absltest

import instruction_manager


class MyTestCase(absltest.TestCase):
    def test_sampling(self):
        # Given an instruction manager...
        manager = instruction_manager.InstructionManager(
            mode='english', instruction_bank_size=3, random_state=1)

        # ...when you take a sample from it...
        output = manager.sample('QUESTION_ANSWERING')
        # ...then you should get an instruction.
        expected_output = 'Answer this legal reasoning question.'
        self.assertEqual(output, expected_output)

    def test_many_samples(self):
        # Given an instruction manager...
        manager = instruction_manager.InstructionManager(
            mode='english', instruction_bank_size=10, random_state=1)

        # ...when you take many samples from it...
        outputs = [manager.sample('QUESTION_ANSWERING') for _ in range(100)]
        # ...they should be different.
        self.assertLen(set(outputs), 10)

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            instruction_manager.InstructionManager(
                mode='dummy', instruction_bank_size=3)


if __name__ == '__main__':
    absltest.main()
