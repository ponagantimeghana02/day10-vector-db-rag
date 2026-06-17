# cache_manager.py

import time


# ==================================
# Cache Storage
# ==================================

cache = {}

cache_stats = {
    "hits": 0,
    "misses": 0
}


# ==================================
# Check Cache
# ==================================

def get_cached_response(query):

    if query in cache:

        cache_stats["hits"] += 1

        return cache[query]

    cache_stats["misses"] += 1

    return None


# ==================================
# Store Cache
# ==================================

def cache_response(
    query,
    answer,
    sources
):

    cache[query] = {

        "answer": answer,

        "sources": sources,

        "timestamp": time.time()

    }


# ==================================
# Remove Cache Entry
# ==================================

def remove_cache(query):

    if query in cache:

        del cache[query]

        return True

    return False


# ==================================
# Clear Entire Cache
# ==================================

def clear_cache():

    cache.clear()

    print(
        "Cache cleared."
    )


# ==================================
# Cache Statistics
# ==================================

def get_cache_stats():

    total = (
        cache_stats["hits"]
        +
        cache_stats["misses"]
    )

    hit_rate = 0

    if total > 0:

        hit_rate = (
            cache_stats["hits"]
            / total
        ) * 100

    return {

        "total_queries": total,

        "hits": cache_stats["hits"],

        "misses": cache_stats["misses"],

        "hit_rate": round(
            hit_rate,
            2
        )
    }


# ==================================
# Show Cache Contents
# ==================================

def show_cache():

    print("\nCached Queries")

    print("-" * 50)

    for query in cache:

        print(query)

        print("-" * 50)


# ==================================
# Cache Size
# ==================================

def cache_size():

    return len(cache)


# ==================================
# Test
# ==================================

if __name__ == "__main__":

    cache_response(

        "What is AI?",

        "Artificial Intelligence...",

        ["sample.txt"]

    )

    result = get_cached_response(
        "What is AI?"
    )

    print(result)

    print(
        get_cache_stats()
    )