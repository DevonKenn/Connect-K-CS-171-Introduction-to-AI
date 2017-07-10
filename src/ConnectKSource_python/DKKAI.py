#Author: Toluwanimi Salako
from collections import defaultdict
import random
import sys
import board_model as boardmodel
import math
import time

team_name = "DKK" #TODO change me
class StudentAI():
	#def oppositePlayer(self,player):
	#	if(player==1):
	#		return -1
	#	return 1
	def oppositePlayer(self):
		if(self.player==1):
			return 2
		return 1

	def copyTwoDList(self,list):
		newL = []
		for i in range(width):
			for j in range(height):
				newL[i][j] = list[i][j]

	def availableMoves(self,model):
		width = model.get_width()
		height = model.get_height()
		spaces = defaultdict(int)
		for i in range(width):

			for j in range(height):
				spaces[(i,j)] = model.get_space(i, j)
				if(model.gravity and model.get_space(i,j) ==0):
					break
		moves = [k for k in spaces.keys() if spaces[k] == 0]
		return moves

	def apply(self,model, move,player):

		return model.place_piece(move,player)

	def maxV(self,model,depth,alpha,beta,best =[],prior_moves = [],moves =[]):
		if(depth <=0 or model.winner()):
			return self.heuristic(model,depth),prior_moves,moves
		val = float("-inf");
		best_move = []
		nm = []
		move = ()
		if(len(best)>0):
			prior_moves.append(best[0])
			the_move_index = moves.index(best[0])
			move = moves.pop(the_move_index)
			nm.append(move)
			val2,best_moves,moves2 = self.minV(self.apply(model,move,self.player),(depth-1),alpha,beta,best[1:],prior_moves,moves)
			
			prior_moves.pop()
			if(val2>val):
				val = val2
				best_move = best_moves
			alpha = max(alpha,val)

			if(alpha>=beta):
				moves.extend(nm)
				return alpha,best_move,moves
		while(len(moves)>0):
			move = moves.pop()

			prior_moves.append(move)
			t = nm  + moves
			val2,best_moves,moves2  = self.minV(self.apply(model,move,self.player),(depth-1),alpha,beta,[],prior_moves,t)
			nm.append(move)
			prior_moves.pop()

			if(val2>val):
				val = val2
				best_move = best_moves
			alpha = max(alpha,val)
			if(alpha >=beta):
				moves.extend(nm)
				return alpha ,best_moves,moves
		moves,nm = nm,moves
		return alpha ,best_move,moves

	def minV(self,model,depth,alpha,beta,best =[], prior_moves = [],moves = []):
		if(depth <=0 or model.winner()):
			return self.heuristic(model,depth),prior_moves,moves

		val = float("inf")
		best_move = []
		nm = []
		move =()
		if(len(best)>0):
			the_move_index = moves.index(best[0])
			move = moves.pop(the_move_index)
			prior_moves.append(move)
			val2,best_moves,moves2 =  self.maxV(self.apply(model,move,self.oppositePlayer()),(depth-1),alpha,beta,best[1:],prior_moves,moves)
			nm.append(move)
			prior_moves.pop()
			if(val2<val):
				val = val2
				best_move = best_moves
			beta = min(beta,val)

			if(beta<=alpha):
				moves.extend(nm)
				return beta,best_moves,moves

		while(len(moves)>0):
			move = moves.pop()

			prior_moves.append(move)
			t = nm + moves
			val2,best_moves,moves2 = self.maxV(self.apply(model,move,self.oppositePlayer()),(depth-1),alpha,beta,[],prior_moves,t)
			nm.append(move)

			prior_moves.pop()
			if(val2<val):
				val = val2
				best_move = best_moves
			beta = min(beta,val)

			if(beta<=alpha):
				moves.extend(nm)
				return beta,best_moves,moves

		nm,moves = moves,nm
		return beta,best_move,moves

	def checkRight(self,model,move,player,second = False):
		x,y = move
		count =0
		if((x-1<0 or model.pieces[x-1][y] != player) ):# and (x-2<0 or model.pieces[x-2][y] !=player)):
			count =1
			while(x+count < model.width and model.pieces[x][y] == model.pieces[x+count][y]):
				if(count>=model.k_length):
					count+=1000
				else:
					count+=1
			#if(count==model.k_length):
			#	count -=1

		secondCount =0
		if(second==False and x+count+1<model.width and model.pieces[x+count][y] ==0 and model.pieces[x][y] ==model.pieces[x+count+1][y]):
			count =1
			secondCount = self.checkRight(model,(x+count+1,y), player,True)
			if(math.sqrt(secondCount) + count >=model.k_length):
				secondCount==1000
		elif (  count<model.k_length and (x-2<0 or (model.pieces[x-2][y] !=0 and model.pieces[x-2][y] !=player ) or  model.pieces[x-1][y] != player and model.pieces[x-1][y] != 0  ) and (x+model.k_length >= model.width or (model.pieces[x+count][y] !=0 and model.pieces[x+count][y] !=player)  ) ):
			return count
		return count*count + secondCount

	def checkDiagUpRight(self,model,move,player,second = False):
		x,y = move
		count =0
		if((x-1<0 or y-1<0 or model.pieces[x-1][y-1] != player) ):#and (x-2<0 or y-2<0 or model.pieces[x-2][y-2]!=player)):
			count =1
			while(x+count < model.width and y+count<model.height and model.pieces[x][y] == model.pieces[x+count][y+count]):
				if(count>=model.k_length):
					count+=1000
				else:
					count+=1
			#if(count==model.k_length):
			#	count -=1
		secondCount = 0
		if(second==False and x+count+1<model.width and y+count+1<model.height and model.pieces[x+count][y+count] ==0 and model.pieces[x][y] ==model.pieces[x+count+1][y+count+1]):
			count =1
			secondCount = self.checkDiagUpRight(model,(x+count+1,y+count+1), player,True)
			if(math.sqrt(secondCount)+ count >=model.k_length):
				secondCount+=1000
		elif ( count<model.k_length and (x-2<0 or y-2<0 or (model.pieces[x-2][y-2] !=player and model.pieces[x-2][y-2] !=0 ) ) and (x+model.k_length>= model.width or y+model.k_length>=model.height or (model.pieces[x+count][y+count] !=0 and model.pieces[x+count][y+count] !=player  ) ) ):
			return count

		return count*count + secondCount

	def checkDiagDownRight(self,model,move,player,second = False):
		x,y = move
		count =0
		if((x-1<0 or y+1>=model.height or model.pieces[x-1][y+1] != player) ):#and (x-2<0 or y+2>=model.height or model.pieces[x-2][y-2] !=player)):
			count =1
			while(x+count < model.width and y-count >=0 and model.pieces[x][y] == model.pieces[x+count][y-count]):
				if(count>=model.k_length):
					count+=1000
				else:
					count+=1
			#if(count==model.k_length):
			#	count -=1

		secondCount =0
		if(second==False and x+count+1<model.width and y-count-1>=0 and model.pieces[x+count][y-count] ==0 and model.pieces[x][y] ==model.pieces[x+count+1][y-count-1]):
			count =1
			secondCount += self.checkDiagDownRight(model,(x+count+1,y-count-1), player,True)
			if(math.sqrt(secondCount)+ count >=model.k_length):
				secondCount+=1000
		elif ( count<model.k_length and (x-2<0 or y+2>=model.height or (model.pieces[x-2][y+2] !=player and model.pieces[x-2][y+2] !=0 ) ) and (x+model.k_length >= model.width or y-model.k_length<0 or (model.pieces[x+count][y-count] !=0 and model.pieces[x+count][y-count] !=player ) ) ):
			return count

		return count*count + secondCount
	def checkUp(self,model,move,player,second = False):
		x,y = move
		#Calculate the head and the tail and from there, if it has (k_length-1) and the head and tail are 0, if depth is max (current turn)
		#then give up, if other depth then high priority (MAYBE SAVE A GLOBAL VARIABLE TO SEE IF IT IS NEW TO THIS DEPTH? give it a turn num
		#attribute?) 
		#Otherwise, if the head and tail are positioned such that there isn't room for this way to win, abandon it with 0 priority
		count =0
		head = False
		tail = False

		if( (y-1<0 or  model.pieces[x][y-1] != player) and (y-2>=0 or  model.pieces[x][y-2] != player) ):
			if(y-1>=0 and model.pieces[x][y-1] ==0):
				head = True
			count =1
			back_count = 1
			while(y+count < model.height and model.pieces[x][y] == model.pieces[x][y+count]):
				count+=1

			if(second):
				return count

			while(y-back_count-1 >=0  and (model.pieces[x][y-back_count-1] ==model.pieces[x][y] or model.pieces[x][y-back_count-1] ==0) ):
				back_count +=1

			if(y+count < model.height and model.pieces[x][y+count] ==0 ):#else tail is false, it ends at an enemy or wall
				tail =True
				if(count == model.k_length -1 and head):#if head and tail then you always win!
					return (10 ** (model.k_length + 4))
			if(count + back_count  < model.k_length and not tail):
				return 0
		else:
			return 0
		secondCount =0
		if(second==False and y+count+1<model.height and model.pieces[x][y+count] ==0 and model.pieces[x][y] ==model.pieces[x][y+count+1]):
			secondCount += self.checkUp(model,(x,y+count+1), player,True)

		return 1



	def heuristic(self,model,depth):
		width = model.get_width()
		height = model.get_height()
		print(height)
		total =0
		enemyTotal =0
		spaces = defaultdict(int)
		moves = []
		enemyMoves = []
		for i in range(width):
			for j in reversed(range(height)):
				if(model.get_space(i,j)==self.player):#NEW
					move = (i,j)
					total = total + self.checkUp(model,move,self.player) + self.checkDiagDownRight(model,move,self.player) +self.checkDiagUpRight(model,move,self.player)+ self.checkRight(model,move,self.player)

				elif(model.get_space(i,j)!=0):#=self.oppositePlayer()):
					move2 = (i,j)
					enemyTotal = enemyTotal+ self.checkUp(model,move2,self.oppositePlayer()) + self.checkDiagDownRight(model,move2,self.oppositePlayer()) + self.checkDiagUpRight(model,move2,self.oppositePlayer()) + self.checkRight(model,move2,self.oppositePlayer())
		print(enemyTotal)
		return total - enemyTotal #- depth

	def miniMax(self,model,depth):

		maxValue = None
		maxMove = None
		moves = self.availableMoves(model)
		best = []
		the_move = None
		alpha = float("-inf")
		move =()
		nm = []
		outerD = depth + 1
		for i in range(1,outerD):
			print(moves)

			if(best and len(moves) >0):
				the_move_index = moves.index(best[0])
				move = moves.pop(the_move_index)
				nm.append(move)
				v,best2,moves2=self.minV(self.apply(model,move,self.player),i,alpha,float("inf"),best[1:],[move],moves)
				if(maxValue==None or v>maxValue):
					maxValue = v
					alpha = v
					maxMove = move
					best = []
					best.extend(best2)
			while(moves):
				move = moves.pop()
				t = moves + nm

				v,best2,moves2=self.minV(self.apply(model,move,self.player),i,alpha,float("inf"),[],[move],t)
				#print(moves2)
				#print(moves)
				#print()
				nm.append(move)
				if(v>alpha):
					maxValue = v
					alpha =v
					maxMove = move
					best = []
					best.extend(best2)
			moves,nm = nm,moves
		
		assert maxValue!=None

		return maxMove
	
	def make_move(self,model, deadline):
		'''Write AI Here. Return a tuple (col, row)'''
		start_time = int(round(time.time() ))

		return self.miniMax(self.model,3)

	def __init__(self, player, state):
		self.last_move = state.get_last_move()
		self.model = state
		self.player = player


'''===================================
DO NOT MODIFY ANYTHING BELOW THIS LINE
==================================='''

is_first_player = False
deadline = 0
model = None
def make_ai_shell_from_input():
	'''
	Reads board state from input and returns the move chosen by StudentAI
	DO NOT MODIFY THIS
	'''
	global is_first_player
	global model
	global deadline
	ai_shell = None
	begin =  "makeMoveWithState:"
	end = "end"

	go = True
	while (go):
		mass_input = input().split(" ")
		if (mass_input[0] == end):
			sys.exit()
		elif (mass_input[0] == begin):
			#first I want the gravity, then number of cols, then number of rows, then the col of the last move, then the row of the last move then the values for all the spaces.
			# 0 for no gravity, 1 for gravity
			#then rows
			#then cols
			#then lastMove col
			#then lastMove row.
			#then deadline.
			#add the K variable after deadline.
			#then the values for the spaces.
			#cout<<"beginning"<<endl;
			gravity = int(mass_input[1])
			col_count = int(mass_input[2])
			row_count = int(mass_input[3])
			last_move_col = int(mass_input[4])
			last_move_row = int(mass_input[5])

			#add the deadline here:
			deadline = -1
			deadline = int(mass_input[6])
			k = int(mass_input[7])
			#now the values for each space.


			counter = 8
			#allocate 2D array.
			model = boardmodel.BoardModel(col_count, row_count, k, gravity)
			count_own_moves = 0

			for col in range(col_count):
				for row in range(row_count):
					model.pieces[col][row] = int(mass_input[counter])
					if (model.pieces[col][row] == 1):
						count_own_moves += model.pieces[col][row]
					counter+=1

			if (count_own_moves % 2 == 0):
				is_first_player = True

			model.last_move = (last_move_col, last_move_row)
			ai_shell = StudentAI(1 if is_first_player else 2, model)

			return ai_shell
		else:
			print("unrecognized command", mass_input)
		#otherwise loop back to the top and wait for proper _input.
	return ai_shell

def return_move(move):
	'''
	Prints the move made by the AI so the wrapping shell can input it
	DO NOT MODIFY THIS
	'''
	made_move = "ReturningTheMoveMade";
	#outputs made_move then a space then the row then a space then the column then a line break.
	print(made_move, move[0], move[1])

def check_if_first_player():
	global is_first_player
	return is_first_player

if __name__ == '__main__':
	'''
	DO NOT MODIFY THIS
	'''
	#global deadline

	print ("Make sure this program is ran by the Java shell. It is incomplete on its own. :")
	go = True
	while (go): #do this forever until the make_ai_shell_from_input function ends the process or it is killed by the java wrapper.
		ai_shell = make_ai_shell_from_input()
		moveMade = ai_shell.make_move(model,deadline)
		return_move(moveMade)
		del ai_shell
		sys.stdout.flush()