import copy
import random


class Auction():

    def __init__(self, value_function, bidder_set: list, object_set: list):
        # 输入
        self.bidder_set = bidder_set  # 竞拍者
        self.object_set = object_set  # 竞拍对象
        self.value_matrix = dict()
        for bidder in self.bidder_set:
            for object in self.object_set:
                tuple_temp = (bidder, object)
                self.value_matrix[tuple_temp] = value_function(bidder, object)
        # 每个竞拍者可竞拍对象集合(默认采用均竞拍)
        self.available_objects_for_bidders = dict()
        for bidder in self.bidder_set:
            self.available_objects_for_bidders[bidder] = copy.deepcopy(self.object_set)
        # ε阈值
        self.epsilon = 1.0

    def bidding(self, current_assignment, current_prices):
        """
        根据最新价格确定下一轮报价
        """
        new_bid = {}
        for bidder in self.bidder_set:
            if bidder in current_assignment:
                continue
            # 未竞拍到物品的继续报价
            else:
                v_ij = []
                available_object_list = copy.deepcopy(self.available_objects_for_bidders[bidder])
                for object in available_object_list:
                    v_ij.append(self.value_matrix[bidder, object] - current_prices[object])
                max_v = max(v_ij)  # 最大收益
                bid_target = available_object_list[v_ij.index(max_v)]  # 最大收益对象
                v_ij.remove(max_v)
                available_object_list.remove(bid_target)
                sub_max_v = max(v_ij)  # 次大收益
                bid_price = self.value_matrix[bidder, bid_target] - sub_max_v + self.epsilon  # 下次报价
                if bid_target in new_bid:
                    new_bid[bid_target][0].append(bidder)
                    new_bid[bid_target][1].append(bid_price)
                else:
                    new_bid[bid_target] = [[bidder], [bid_price]]
        return new_bid

    def assigning(self, bid_prices, current_assignment, current_prices):
        """
        根据最新报价重新分配竞拍结果
        """
        new_prices = {}
        for object in self.object_set:
            # 收到报价的物品重新竞拍
            if object in bid_prices:
                bidders, prices = bid_prices[object]
                max_price = max(prices)
                bidder=bidders[prices.index(max_price)]
                new_prices[object] = max_price
                for people, object_ in current_assignment.items():
                    if object_ == object:
                        current_assignment.pop(people)
                        current_assignment[bidder] = object
                        break
                if bidder not in current_assignment:
                    current_assignment[bidder] = object
            else:
                new_prices[object] = current_prices[object]
        return new_prices, current_assignment

    def run(self, test_flag=False):
        """
        运行拍卖
        """
        # 初始竞拍结果(满足ε互补松弛条件)
        current_assignment = dict()
        # current_assignment[self.bidder_set[0]] = self.object_set[0]
        # current_assignment[self.bidder_set[1]] = self.object_set[1]
        # 初始报价
        current_prices = dict()
        for object in self.object_set:
            current_prices[object] = 0
        # 竞拍过程
        k = 1
        while True:
            new_bid = self.bidding(current_assignment, current_prices)
            if len(new_bid) > 0:
                new_price, new_assignment = self.assigning(new_bid, current_assignment, current_prices)
                current_assignment = new_assignment
                current_prices = new_price
            else:
                break
            k += 1
            if test_flag:
                print("第{}次竞拍：".format(k))
                print("\t竞拍结果为:")
                print("\t", current_assignment)
                print("\t最新定价为:")
                print("\t", current_prices)
        return current_assignment


if __name__ == "__main__":
    def value_test(a, b):
        return random.randint(1, 20)
    # 竞拍者集合
    bidder_set = ['A1', 'A2', 'A3', 'A4']
    # 竞拍对象集合
    object_set = ['B1', 'B2', 'B3', 'B4']
    auction = Auction(value_test, bidder_set, object_set)
    auction.run(True)
