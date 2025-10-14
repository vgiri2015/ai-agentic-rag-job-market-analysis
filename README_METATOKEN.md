# MetaToken Semantic Cache Integration

This integration routes your LLM calls through MetaToken's semantic cache service to reduce costs.

## Setup

1. **Install dependencies:**
```bash
pip install requests
```

2. **Set environment variable:**
```bash
export METATOKEN_USER_TOKEN="mtk_metatoken_aRW48mvv58KpWc5hlMH_BA"
```

3. **Import and use:**
```python
from metatoken_cache import MetaTokenCache

cache = MetaTokenCache()
result = cache.query("Your prompt here")
```

## Features

✓ Automatic semantic caching  
✓ 40-80% cost reduction  
✓ Transparent integration  
✓ Real-time analytics in MetaToken dashboard  

## Configuration

Configure cache settings in your MetaToken dashboard:
- Similarity threshold
- Cache TTL
- Enable/disable caching

## Support

Visit MetaToken dashboard for analytics and support.
