"""
This program visualize the generated trace on the given game.
"""
from PIL import Image
import numpy as np

class HuntVisualizer(object):
    def __init__(self) -> None:
        self.agent = Image.open('.\\Visualization\\Hunt\\Agent_nofood.png').resize((80,80))
        self.agent_back = Image.open('.\\Visualization\\Hunt\\Agent.png').resize((80,80))
        self.lion = Image.open('.\\Visualization\\Hunt\\lion.png').resize((100,100))
        self.agent_channels = self.agent.split()
        self.background = Image.open(
            '.\\Visualization\\Hunt\\Background.png').resize((800, 600))
        self.mid_frame_num = 10
        self.n_agents = 3
        self.agent_locs = [[(220,250),(700,300)],[(220, 325), (500,100)],[(320, 300),(700,300)]]
        self.lion_locs = [(700,200),(500,250)]
        self.lion_xy = []
        self.frame = self.background.copy()
        self.anime = []
        self.states = []
        self.states_xy = []
        self.agent_use = []

    def init_figs(self):
        # S0 0,0,0
        self.reset()
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][0],self.agent_locs[2][0]))
        self.agent_use.append((0,0,0))
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.frame.paste(self.lion,self.lion_locs[0],mask=self.lion.split()[3])
        self.lion_xy.append(self.lion_locs[0])
        #self.frame.show()
        self.states.append(self.frame.copy())
        # S1 0,0,1
        self.reset()
        self.agent_use.append((0,0,1))
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][0],self.agent_locs[2][1]))
        self.lion_xy.append(self.lion_locs[0])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.frame.paste(self.lion,self.lion_locs[0],mask=self.lion.split()[3])
        self.states.append(self.frame.copy())
        # S2 0,1,0
        self.reset()
        self.agent_use.append((0,1,0))
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][1],self.agent_locs[2][0]))
        self.lion_xy.append(self.lion_locs[0])
        self.frame.paste(self.lion,self.lion_locs[0],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        # S3 0,1,1
        self.reset()
        self.agent_use.append((0,1,1))
        self.lion_xy.append(self.lion_locs[1])
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][1],self.agent_locs[2][1]))
        self.frame.paste(self.lion,self.lion_locs[1],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        # S4 1,0,0
        self.agent_use.append((1,0,0))
        self.reset()
        self.states_xy.append((self.agent_locs[0][1],self.agent_locs[1][0],self.agent_locs[2][0]))
        self.lion_xy.append(self.lion_locs[0])
        self.frame.paste(self.lion,self.lion_locs[0],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        # S5 1,0,1
        self.reset()
        self.agent_use.append((1,0,1))
        self.states_xy.append((self.agent_locs[0][1],self.agent_locs[1][0],self.agent_locs[2][1]))
        self.lion_xy.append(self.lion_locs[1])
        self.frame.paste(self.lion,self.lion_locs[1],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        # S6 1,1,0
        self.agent_use.append((1,1,0))
        self.reset()
        self.states_xy.append((self.agent_locs[0][1],self.agent_locs[1][1],self.agent_locs[2][0]))
        self.lion_xy.append(self.lion_locs[1])
        self.frame.paste(self.lion,self.lion_locs[1],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        # S7 1,1,1
        self.agent_use.append((1,1,1))
        self.reset()
        self.states_xy.append((self.agent_locs[0][1],self.agent_locs[1][1],self.agent_locs[2][1]))
        self.lion_xy.append(self.lion_locs[1])
        self.frame.paste(self.lion,self.lion_locs[1],mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
    def reset(self):
        self.frame=self.background.copy()

    def process(self,from_s,to_s):
        ori_xy = np.array(self.states_xy[from_s])
        des_xy = np.array(self.states_xy[to_s])
        mid_xy = []
        dis_xy = [des_xy[0] - ori_xy[0], des_xy[1] - ori_xy[1], des_xy[2] - ori_xy[2]]
        ori_lion = np.array(self.lion_xy[from_s])
        des_lion = np.array(self.lion_xy[to_s])
        dis_lion = np.array([des_lion[0]-ori_lion[0],des_lion[1]-ori_lion[1]])
        mid_lion_xy = []
        for i in range(self.mid_frame_num):
            mid_lion_xy.append(ori_lion+dis_lion*i/self.mid_frame_num)
        for i in range(self.mid_frame_num):
            mid_xy.append((ori_xy[0] + dis_xy[0]*i / self.mid_frame_num,
                           ori_xy[1] + dis_xy[1]*i / self.mid_frame_num,
                           ori_xy[2] + dis_xy[2]*i / self.mid_frame_num))
        self.reset()
        agents = []
        self.frame.paste(self.lion,tuple(ori_lion),mask=self.lion.split()[3])
        for i in range(self.n_agents):
            if self.agent_use[from_s][i] == 1:
                agents.append(self.agent_back)
            else:
                agents.append(self.agent)

            self.frame.paste(agents[i],
                            self.states_xy[from_s][i],
                            mask=agents[i].split()[3])
        self.anime.append(self.frame)

        for i in range(self.mid_frame_num):
        #for xy in mid_xy:
            xy=mid_xy[i]
            self.reset()
            for j in range(self.n_agents):
                self.frame.paste(agents[j],
                            (int(xy[j][0]),int(xy[j][1])),
                            mask=agents[j].split()[3])
            #self.frame.show()
            self.frame.paste(self.lion,(int(mid_lion_xy[i][0]),int(mid_lion_xy[i][1])),mask=self.lion.split()[3])
            self.anime.append(self.frame)


        self.reset()
        self.frame.paste(self.lion,tuple(des_lion),mask=self.lion.split()[3])
        for i in range(self.n_agents):
            self.frame.paste(agents[i],
                            self.states_xy[to_s][i],
                            mask=agents[i].split()[3])

        
        self.anime.append(self.frame)

    def save_gif(self,task_name):
        self.anime[0].save(task_name + '.gif',
                           save_all=True,
                           append_images=self.anime,
                           duration=10)

class GambleVisualizer(object):
    def __init__(self):
        self.agent = Image.open(
            '.\\Visualization\\FairGamble\\Person.png').resize((90, 90))
        self.agent_channels = self.agent.split()
        self.background = Image.open(
            '.\\Visualization\\FairGamble\\Background.png').resize((800, 600))
        self.mid_frame_num = 10
        self.frame = self.background.copy()
        self.anime = []
        self.states = []
        self.states_xy = []

    def init_figs(self):
        self.frame.paste(self.agent, (20, 500), mask=self.agent_channels[3])
        self.frame.paste(self.agent, (40, 200), mask=self.agent_channels[3])
        self.states_xy.append(((20, 50), (20, 200)))
        self.states.append(self.frame.copy())

        self.reset()
        self.frame.paste(self.agent, (400, 50), mask=self.agent_channels[3])
        self.frame.paste(self.agent, (400, 150), mask=self.agent_channels[3])
        self.states_xy.append(((400, 50), (400, 150)))
        self.states.append(self.frame.copy())

        self.reset()
        self.frame.paste(self.agent, (600, 215), mask=self.agent_channels[3])
        self.frame.paste(self.agent, (600, 315), mask=self.agent_channels[3])
        self.states_xy.append(((600, 215), (600, 315)))
        self.states.append(self.frame.copy())

        self.reset()
        self.frame.paste(self.agent, (400, 450), mask=self.agent_channels[3])
        self.frame.paste(self.agent, (400, 350), mask=self.agent_channels[3])
        self.states_xy.append(((400, 450), (400, 350)))
        self.states.append(self.frame.copy())

    def reset(self):
        self.frame = self.background.copy()

    def process(self, from_s, to_s):
        ori_xy = np.array(self.states_xy[from_s])
        des_xy = np.array(self.states_xy[to_s])
        mid_xy = []
        dis_xy = [des_xy[0] - ori_xy[0], des_xy[1] - ori_xy[1]]
        for i in range(self.mid_frame_num):
            mid_xy.append((ori_xy[0] + dis_xy[0]*i / self.mid_frame_num,
                           ori_xy[1] + dis_xy[1]*i / self.mid_frame_num))
        # Push to anime
        self.reset()
        self.frame.paste(self.agent,
                         self.states_xy[from_s][0],
                         mask=self.agent_channels[3])
        self.frame.paste(self.agent,
                         self.states_xy[from_s][1],
                         mask=self.agent_channels[3])
        self.anime.append(self.frame)
        for xy in mid_xy:
            self.reset()
            self.frame.paste(self.agent,
                             (int(xy[0][0]),int(xy[0][1])),
                             mask=self.agent_channels[3])
            self.frame.paste(self.agent,
                             (int(xy[1][0]),int(xy[1][1])),
                             mask=self.agent_channels[3])
            self.anime.append(self.frame)
        self.reset()
        self.frame.paste(self.agent,
                         self.states_xy[to_s][0],
                         mask=self.agent_channels[3])
        self.frame.paste(self.agent,
                         self.states_xy[to_s][1],
                         mask=self.agent_channels[3])
        self.anime.append(self.frame)

    def save_gif(self, task_name):
        self.anime[0].save(task_name + '.gif',
                           save_all=True,
                           append_images=self.anime,
                           duration=10)


class CaEVisualizer(object):
    def __init__(self) -> None:
        self.agent = Image.open('.\\Visualization\\CaE\\Agent_nofood.png').resize((80,80))
        self.agent_channels = self.agent.split()
        self.background = Image.open(
            '.\\Visualization\\CaE\\Background.png').resize((800, 600))
        self.mid_frame_num = 10
        self.n_agents = 3
        self.agent_locs = [[(200,200),(-80,300)],[(300,300), (-80,300)],[(400,200),(360,600)]]
        self.frame = self.background.copy()
        self.anime = []
        self.states = []
        self.states_xy = []
        self.agent_use = []

    def init_figs(self):
        # S0 0,0,0
        self.reset()
        #self.frame.show()
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][0],self.agent_locs[2][0]))
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())
        
        # S1 0,0,1
        self.reset()
        #self.frame.show()
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][0],self.agent_locs[2][1]))
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())

        # S2 0,1,0
        self.reset()
        #self.frame.show()
        self.states_xy.append((self.agent_locs[0][0],self.agent_locs[1][1],self.agent_locs[2][0]))
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())

        # S3 1,0,0
        self.reset()
        #self.frame.show()
        self.states_xy.append((self.agent_locs[0][1],self.agent_locs[1][0],self.agent_locs[2][0]))
        for i in range(self.n_agents):
            self.frame.paste(self.agent, self.states_xy[-1][i], mask=self.agent_channels[3])
        self.states.append(self.frame.copy())

    def reset(self):
        self.frame=self.background.copy()

    def process(self,from_s,to_s):
        ori_xy = np.array(self.states_xy[from_s])
        des_xy = np.array(self.states_xy[to_s])
        mid_xy = []
        dis_xy = [des_xy[0] - ori_xy[0], des_xy[1] - ori_xy[1], des_xy[2] - ori_xy[2]]
        for i in range(self.mid_frame_num):
            mid_xy.append((ori_xy[0] + dis_xy[0]*i / self.mid_frame_num,
                           ori_xy[1] + dis_xy[1]*i / self.mid_frame_num,
                           ori_xy[2] + dis_xy[2]*i / self.mid_frame_num))
        self.reset()
        agents = []
        for i in range(self.n_agents):
            agents.append(self.agent)
            self.frame.paste(agents[i],
                            self.states_xy[from_s][i],
                            mask=agents[i].split()[3])
        self.anime.append(self.frame)


        for xy in mid_xy:
            self.reset()
            for i in range(self.n_agents):
                self.frame.paste(agents[i],
                            (int(xy[i][0]),int(xy[i][1])),
                            mask=agents[i].split()[3])
            #self.frame.show()
            self.anime.append(self.frame)


        self.reset()
        for i in range(self.n_agents):
            self.frame.paste(agents[i],
                            self.states_xy[to_s][i],
                            mask=agents[i].split()[3])

        
        self.anime.append(self.frame)
        pass

    def save_gif(self,task_name):
        self.anime[0].save(task_name + '.gif',
                           save_all=True,
                           append_images=self.anime,
                           duration=10)

if __name__ == "__main__":
    vis = HuntVisualizer()
    vis.init_figs()
    trace = np.load('HuntVT10Tracebefore.npy')
    for i in range(len(trace)-1):
        start_s,to_s = trace[i],trace[i+1]
        vis.process(start_s,to_s)
    vis.save_gif('HuntVT10Tracebefore')
    trace = np.load('HuntVT10Traceafter.npy')
    for i in range(len(trace)-1):
        start_s,to_s = trace[i],trace[i+1]
        vis.process(start_s,to_s)
    vis.save_gif('HuntVT10Traceafter')
    # vis = CaEVisualizer()
    # vis.init_figs()
    # vis.process(0,1)
    # vis.process(1,2)
    # vis.process(2,0)
    # vis.save_gif('CaE')

    vis = HuntVisualizer()
    vis.init_figs()
    vis.process(0,1)
    vis.process(1,3)
    vis.process(3,0)
    vis.save_gif('Hunt')
