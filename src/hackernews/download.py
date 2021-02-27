"""
adapted from https://github.com/mehmetkose/tangrowth/blob/master/async_crawler.py
"""
import json
import time
import aiohttp
import asyncio
import async_timeout
import requests
import pymongo


class HackerNewsItems:

    URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self):
        self.downloaded_items = set()
        self.work_queue = asyncio.Queue()
        self.expand_relation = False
        self.mongo = pymongo.mongo_client.MongoClient()
        # self.mongo.drop_database("hackernews")
        self.db = self.mongo.get_database("hackernews")
        self.db_items = self.db.get_collection("items")
        self._last_verbose_time = 0
        self._last_verbose_items = 0

    def num_live_items(self):
        return int(requests.get(f"{self.URL}/maxitem.json").text)

    def download(self, start_item: int, end_item: int, num_tasks: int = 10):
        for i in range(start_item, end_item):
            self.work_queue.put_nowait(i)

        tasks = [self.handle_task(task_id) for task_id in range(num_tasks)]

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    async def get_item(self, index):
        url = f"{self.URL}/item/{index}.json"

        async with aiohttp.ClientSession() as session:
            try:
                with async_timeout.timeout(10):
                    # print("GET", url)
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            return {'error': response.status}
            except Exception as err:
                return {'error': err}

    async def handle_task(self, task_id):

        while not self.work_queue.empty():
            item_index = await self.work_queue.get()
            if item_index not in self.downloaded_items:
                item = await self.get_item(item_index)
                self.downloaded_items.add(item_index)
                if not item:
                    print("EMPTY", item_index, item)
                elif "error" in item:
                    print("ERROR", item_index, item)
                else:
                    # print("ITEM", item_index, item)
                    self.store_item(item)

                    if self.expand_relation:
                        if item.get("kids"):
                            for kid_index in item["kids"]:
                                if kid_index not in self.downloaded_items:
                                    self.work_queue.put_nowait(kid_index)

            if task_id == 0:
                cur_time = time.time()
                if cur_time - self._last_verbose_time > 1:
                    num_items = len(self.downloaded_items)
                    num_per_sec = (num_items - self._last_verbose_items) / (cur_time - self._last_verbose_time)
                    print(f"downloaded/stored: {num_items} items ({num_per_sec:0.2f}/sec)")

                    self._last_verbose_items = num_items
                    self._last_verbose_time = cur_time

    def store_item(self, item):
        item["_id"] = item["id"]
        if self.db_items.update_one({"_id": item["id"]}, {"$set": item}).matched_count == 0:
            self.db_items.insert_one(item)


if __name__ == "__main__":

    api = HackerNewsItems()
    api.download(0, 10000, num_tasks=10)

    print(json.dumps(api.downloaded_items, indent=2))
    print(len(api.downloaded_items))