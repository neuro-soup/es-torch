import pytest
import torch

from es_torch.fitness_shaping import centered_rank


class TestCenteredRank:
    """Test centered rank transform with various cases including ties."""
    
    @pytest.mark.parametrize("rewards,expected", [
        # No ties
        ([1.0, 2.0, 3.0, 4.0], [-0.5, -1/6, 1/6, 0.5]),
        # All ties
        ([2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]),
        # Some ties
        ([1.0, 2.0, 2.0, 3.0], [-0.5, 0.0, 0.0, 0.5]),
        # Multiple tie groups
        ([1.0, 1.0, 2.0, 2.0, 2.0, 3.0], [-0.4, -0.4, 0.1, 0.1, 0.1, 0.5]),
        # Negative values with ties
        ([-2.0, -1.0, -1.0, 0.0, 1.0], [-0.5, -0.125, -0.125, 0.25, 0.5]),
        # Empty tensor
        ([], []),
        # Single value
        ([5.0], [0.0]),
        # Two values
        ([1.0, 2.0], [-0.5, 0.5]),
    ])
    def test_rank_transform(self, rewards, expected):
        rewards_tensor = torch.tensor(rewards)
        expected_tensor = torch.tensor(expected) if expected else torch.tensor([])
        
        ranks = centered_rank(rewards_tensor)
        
        assert torch.allclose(ranks, expected_tensor, atol=1e-6)
    
    def test_large_tensor_many_ties(self):
        """Test with large tensor containing many tied values."""
        rewards = torch.tensor([1.0] * 10 + [2.0] * 10 + [3.0] * 10)
        ranks = centered_rank(rewards)
        
        # First 10: avg rank 4.5, centered -10, normalized -10/29
        assert torch.allclose(ranks[:10], torch.full((10,), -10/29), atol=1e-6)
        # Middle 10: avg rank 14.5, centered 0, normalized 0
        assert torch.allclose(ranks[10:20], torch.zeros(10), atol=1e-6)
        # Last 10: avg rank 24.5, centered 10, normalized 10/29
        assert torch.allclose(ranks[20:], torch.full((10,), 10/29), atol=1e-6)
    
    def test_output_range(self):
        """Test that output is always in [-0.5, 0.5]."""
        for _ in range(10):
            rewards = torch.randn(20)
            ranks = centered_rank(rewards)
            assert ranks.min() >= -0.5 - 1e-6
            assert ranks.max() <= 0.5 + 1e-6