from connect_4_game import connect_4_game
from connect_4_nn import connect_4_nn
from minimax_player import minimax_player
from player import player
import sys
import datetime

class simple_c4_trainer():
    """
    The simple trainer plays a training_nn against itself num_train times to train, and then compares it to the best_nn so far.
    If training_nn wins threshold% of the num_test test games, then training_nn becomes best_nn.
    """
    def __init__(self, best_nn_checkpoint: str) -> None:
        self.best_nn_checkpoint = best_nn_checkpoint
        
    def test_best_nn(self, test_player: player, rounds: int = 10, clear_report: bool = True) -> None:
        """
        This plays this trainers best_nn against player rounds times.
        It then puts the report in simple_c4_trainer_reports.txt.
        It clears the report first if clear_report.
        """
        best_wins = 0
        test_wins = 0
        draws = 0

        nn = connect_4_nn(self.best_nn_checkpoint, name='Best connect 4 nn', expect_partial=True)
        nn_player = minimax_player(lambda x,y: nn(x,y), name='Best nn player')

        # Loop through the test rounds switching who is black and red each time
        for i in range(rounds):
            if i % 2 == 0:
                game = connect_4_game(nn_player, test_player)
            else:
                game = connect_4_game(test_player, nn_player)
        
            game.play(False)

            if game.winner == nn_player:
                best_wins += 1
            elif game.winner == test_player:
                test_wins += 1
            else:
                draws += 1
        
        # print a report of the test
        og = sys.stdout
            
        with open('simple_c4_trainer_report.txt', 'a') as f:
            if clear_report: f.truncate(0)
            sys.stdout = f
            print('--- Testing best connect 4 nn report at {0} ---'.format(datetime.datetime.now().strftime("%H:%M:%S")))
            print('\tBest nn wins: {0}'.format(best_wins))
            print('\t{1} wins: {0}'.format(test_wins, test_player.name))
            print('\tDraws: {0}'.format(draws))
            sys.stdout = og

    def train(self, rounds: int = 1, threshold: float = 0.55, num_test: int = 10, num_train: int = 100, batch_size = 1, epochs = 5) -> None:
        # We will do rounds rounds of making a new training_nn, training it against itself,
        # and then comparing it to our best_nn. 
        for j in range(rounds):
            training_nn = connect_4_nn(checkpoint_name=self.best_nn_checkpoint, name='Training connect 4 nn')
            best_nn = connect_4_nn(checkpoint_name=self.best_nn_checkpoint, name='Best connect 4 nn', expect_partial=True)

            training_nn.train(num_train)

            trained_wins = 0
            best_wins = 0

            for i in range(num_test):
                if i % 2 == 0:
                    player1 = minimax_player(lambda x,y: training_nn(x,y), 1, name = 'train')
                    player2 = minimax_player(lambda x,y: best_nn(x,y), 1, name = 'best')
                else:
                    player1 = minimax_player(lambda x,y: best_nn(x,y), 1, name = 'best')
                    player2 = minimax_player(lambda x,y: training_nn(x,y), 1, name = 'train')      

                game = connect_4_game(player1, player2)

                game.play(False)

                if game.board.description == 'Black wins.':
                    if player1.name == 'train':
                        trained_wins += 1
                    else:
                        best_wins += 1
                elif game.board.description == 'Red wins.':
                    if player1.name == 'train':
                        best_wins += 1
                    else:
                        trained_wins += 1

            if (trained_wins / num_test) >= threshold:
                training_nn.save(self.best_nn_checkpoint)
                print('New best network!')
            else:
                best_nn.save(self.best_nn_checkpoint)