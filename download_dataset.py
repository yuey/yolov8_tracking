from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/Users/shailt/Documents/Career\ Files/Stanford_MS_Mechanical\ Engineering/Stanford\ Year\ Two/Spring\ Quarter\ 2023/CS231N_Li/Final\ Project")
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])