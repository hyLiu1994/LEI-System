import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useTable, usePagination } from 'react-table';
import { 
  AppBar, Toolbar, Typography, Container, Paper, Table, 
  TableBody, TableCell, TableContainer, TableHead, TableRow,
  TablePagination, Box, ThemeProvider, createTheme, useMediaQuery,
  Button
} from '@mui/material';
import { 
  MainContainer, ChatContainer, MessageList, Message, 
  MessageInput, ConversationHeader
} from '@chatscope/chat-ui-kit-react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
import { AttachFile as AttachFileIcon } from '@mui/icons-material';

// Fix Leaflet's default icon path issues
let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

ChartJS.register(ArcElement, Tooltip, Legend);

// 一个自定义主题，增大字体大小
const theme = createTheme({
  typography: {
    fontSize: 16, // 增加基础字体大小
    h6: {
      fontSize: '1.5rem', // 增加标题字体大小
    },
  },
  components: {
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: '12px 16px', // 单元格内边距
          fontSize: '1rem', // 增加表格字体大小
        },
        head: {
          fontWeight: 'bold',
          fontSize: '1.1rem', // 增加表头字体大小
        },
      },
    },
    MuiTablePagination: {
      styleOverrides: {
        root: {
          overflow: 'hidden',
          minHeight: '64px', // 增加最小高度
        },
        toolbar: {
          minHeight: '64px', // 增加工具栏的最小高度
          alignItems: 'center', // 确保内容垂直居中
        },
        selectLabel: {
          margin: 0,
          fontSize: '1rem', // 增加标签字体大小
        },
        select: {
          fontSize: '1rem', // 增加选框字体大小
        },
        displayedRows: {
          margin: 0,
          fontSize: '1rem', // 增加显示行数字体大小
        },
      },
    },
  },
});

// 增大聊天组件体大小
const chatStyles = `
  .cs-message__content {
    fontSize: 1rem;
  }
  .cs-message-input__content-editor-wrapper {
    fontSize: 1rem;
  }
  .cs-conversation-header__user-name {
    fontSize: 1.2rem;
  }
  .cs-message-input__content-editor {
    fontSize: 1rem;
  }
`;

function App() {
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]); // 新增状态来存储列名
  const [messages, setMessages] = useState([]); // 添加这行
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('md'));
  const [splitPosition, setSplitPosition] = useState(70); // 默认表格70%
  const [pieData, setPieData] = useState(null);
  const [file, setFile] = useState(null);

  const fetchData = async () => {
    try {
      const response = await fetch('/api/data', {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Received data from server:', result);
      if (result.columns && result.data) {
        setColumns(result.columns);
        setData(result.data);
      } else {
        console.error('Unexpected data structure:', result);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const tableColumns = useMemo(
    () => {
      if (columns.length === 0) {
        return [{
          Header: 'No Data',
          accessor: 'noData',
        }];
      }
      return [
        {
          Header: '#',
          id: 'rowIndex',
          Cell: ({row}) => <div>{row.index + 1}</div>,
        },
        ...columns.map(column => ({
          Header: column,
          accessor: column,
        }))
      ];
    },
    [columns]
  );

  console.log('Table columns:', tableColumns); // 添加这行

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    prepareRow,
    page,
    gotoPage,
    setPageSize,
    state: { pageIndex, pageSize },
  } = useTable(
    {
      columns: tableColumns, // 使用新的 tableColumns
      data,
      initialState: { pageIndex: 0, pageSize: 50 }, // 增加默认显示行数
    },
    usePagination
  );

  console.log('Table data:', data); // 添加这行
  console.log('Table page:', page); // 添加这行

  const handleChangePage = (event, newPage) => {
    gotoPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setPageSize(Number(event.target.value));
  };

  const handleSend = (message) => {
    const newMessage = {
      message,
      sentTime: "just now",
      sender: "user",
      direction: "outgoing",
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    fetch('/api/chat', {  // 使用相对路径
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ "message": message, "session_id": 1}),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Received data:', data);
        let botMessage;
        if (data.type === 'trajectory') {
          console.log('Rendering trajectory:', data.path);
          botMessage = {
            type: 'trajectory',
            message: data.text,
            text: data.text,
            path: data.path,
            center: data.center,
            zoom: data.zoom,
            sender: "bot",
            direction: "incoming",
            sentTime: "just now"
          };
        } else if (data.type === 'image') {
          console.log('Rendering image message:', data.image_url);
          botMessage = {
            type: 'image',
            message: data.text,
            image_url: data.image_url,
            sender: "bot",
            direction: "incoming",
            sentTime: "just now"
          };
        } else {
          console.log('Rendering text message:', data.response);
          botMessage = {
            message: data.response,
            sentTime: "just now",
            sender: "bot",
            direction: "incoming",
          };
        }
        console.log('Bot message object:', botMessage);
        setMessages((prevMessages) => [...prevMessages, botMessage]);

        // 如果响应中包含新的数据，更新表格
        if (data.data) {
          setColumns(data.data.columns);
          setData(data.data.data);
        }
      })
      .catch(error => console.error('Error:', error));
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (e) => {
    const newPosition = (e.clientX / window.innerWidth) * 100;
    setSplitPosition(newPosition);
  };

  const handleMouseUp = () => {
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  const handleAttachClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';
    input.onchange = (event) => {
      const file = event.target.files[0];
      if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/upload', {
          method: 'POST',
          body: formData,
        })
          .then(response => response.json())
          .then(result => {
            console.log('File uploaded successfully:', result);
            // 添加一条消息到聊天界面
            const newMessage = {
              message: `File "${file.name}" uploaded successfully.`,
              sentTime: "just now",
              sender: "system",
              direction: "incoming",
            };
            setMessages((prevMessages) => [...prevMessages, newMessage]);
            // 重新加载数据
            fetchData();
          })
          .catch(error => {
            console.error('Error uploading file:', error);
            // 添加一条错误消息到聊天界面
            const errorMessage = {
              message: `Error uploading file: ${error.message}`,
              sentTime: "just now",
              sender: "system",
              direction: "incoming",
            };
            setMessages((prevMessages) => [...prevMessages, errorMessage]);
          });
      }
    };
    input.click();
  };

  return (
    <ThemeProvider theme={theme}>
      <style>{chatStyles}</style>
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <AppBar position="static">
          <Toolbar sx={{ justifyContent: 'space-between' }}>
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontSize: '1.2rem', 
                textAlign: 'left',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                flexGrow: 1, // 让标题占据剩余空间
                marginRight: 2 // 在标题和logo之间添加一些间距
              }}
            >
              Lightweight, Extensible, and Intelligent Trajectory Data Analysis and Management System
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <img src="/static/logo1.png" alt="Logo 1" style={{ height: 40, marginLeft: 10 }} />
              <img src="/static/logo2.png" alt="Logo 2" style={{ height: 40, marginLeft: 10 }} />
              <img src="/static/logo3.png" alt="Logo 3" style={{ height: 40, marginLeft: 10 }} />
            </Box>
          </Toolbar>
        </AppBar>
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: isSmallScreen ? 'column' : 'row', p: 2, gap: 2, overflow: 'hidden' }}>
          <Paper sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', width: isSmallScreen ? '100%' : `${splitPosition}%` }}>
            <TableContainer sx={{ flexGrow: 1, overflow: 'auto' }}>
              <Table {...getTableProps()} aria-label="data table" stickyHeader size="medium">
                <TableHead>
                  {headerGroups.map(headerGroup => (
                    <TableRow key={headerGroup.id} {...headerGroup.getHeaderGroupProps()}>
                      {headerGroup.headers.map(column => (
                        <TableCell key={column.id} {...column.getHeaderProps()}>
                          {column.render('Header')}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableHead>
                <TableBody {...getTableBodyProps()}>
                  {page.length > 0 ? (
                    page.map(row => {
                      prepareRow(row)
                      return (
                        <TableRow key={row.id} {...row.getRowProps()}>
                          {row.cells.map(cell => (
                            <TableCell key={cell.column.id} {...cell.getCellProps()}>
                              {cell.render('Cell')}
                            </TableCell>
                          ))}
                        </TableRow>
                      )
                    })
                  ) : (
                    <TableRow>
                      <TableCell colSpan={columns.length || 1} align="center">
                        No data available
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              rowsPerPageOptions={[25, 50, 100]}
              component="div"
              count={data.length}
              rowsPerPage={pageSize}
              page={pageIndex}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              sx={{ 
                borderTop: '1px solid rgba(224, 224, 224, 1)',
                '.MuiToolbar-root': {
                  minHeight: '64px', // 再次确保工具栏的最小高度
                },
              }}
            />
          </Paper>
          {!isSmallScreen && (
            <Box
              sx={{
                width: '10px',
                backgroundColor: 'grey.300',
                cursor: 'col-resize',
                '&:hover': { backgroundColor: 'primary.main' },
              }}
              onMouseDown={handleMouseDown}
            />
          )}
          <Paper sx={{ 
            width: isSmallScreen ? '100%' : `calc(${100 - splitPosition}% - 10px)`, 
            display: 'flex', 
            flexDirection: 'column',
            minWidth: '400px', // 增加最小宽度
            maxHeight: '100%', // 添加最大高度
          }}>
            <MainContainer style={{ height: '100%' }}> {/* 确保 MainContainer 占满整个高度 */}
              <ChatContainer style={{ height: '100%' }}> {/* 确保 ChatContainer 占满整个高度 */}
                <ConversationHeader>
                  <ConversationHeader.Content userName="Chat with your data" info="Please upload your data first" />
                </ConversationHeader>
                <MessageList style={{ flexGrow: 1, overflowY: 'auto' }}> {/* 允许消息列表滚动 */}
                  {messages.map((m, i) => (
                    <Message key={i} model={{
                      message: m.type === 'trajectory' || m.type === 'image' ? m.text : m.message,
                      sentTime: m.sentTime,
                      sender: m.sender,
                      direction: m.direction,
                      position: "normal"
                    }}>
                      {m.type === 'trajectory' && (
                        <Message.CustomContent>
                          <div style={{ 
                            height: '500px', // 修改为固定高度 500px
                            width: '500px',  // 修改为固定宽度 500px
                            margin: '10px 0 10px auto',
                            padding: '0',
                            overflow: 'hidden',
                            borderRadius: '8px',
                            display: 'flex',
                            justifyContent: 'flex-end'
                          }}>
                            <div style={{
                              height: '100%',
                              width: '100%',
                            }}>
                              <MapContainer 
                                key={`map-${i}`} 
                                center={m.center} 
                                zoom={m.zoom} 
                                style={{ height: '100%', width: '100%' }}
                                whenCreated={(mapInstance) => {
                                  console.log('Map created:', mapInstance);
                                  setTimeout(() => {
                                    mapInstance.invalidateSize();
                                  }, 100);
                                }}
                              >
                                <TileLayer
                                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                />
                                <Polyline positions={m.path} color="red" />
                                {m.path.map((position, index) => (
                                  <Marker key={index} position={position}>
                                    <Popup>{index === 0 ? 'Start' : index === m.path.length - 1 ? 'End' : `Waypoint ${index}`}</Popup>
                                  </Marker>
                                ))}
                              </MapContainer>
                            </div>
                          </div>
                        </Message.CustomContent>
                      )}
                      {m.type === 'image' && (
                        <Message.CustomContent>
                          <div style={{ 
                            maxWidth: '100%',
                            maxHeight: '100%',
                            overflow: 'hidden',
                            borderRadius: '8px',
                            margin: '10px 0'
                          }}>
                            <img 
                              src={m.image_url} 
                              alt="Bot response" 
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain'
                              }}
                            />
                          </div>
                        </Message.CustomContent>
                      )}
                    </Message>
                  ))}
                </MessageList>
                <MessageInput 
                  placeholder="Type message here" 
                  onSend={handleSend} 
                  attachButton={true}
                  attachDisabled={false}
                  onAttachClick={handleAttachClick}
                  style={{ 
                    flexGrow: 1,
                    border: '1px solid #e0e0e0',
                    borderRadius: '4px',
                    padding: '8px'
                  }} 
                />
              </ChatContainer>
            </MainContainer>
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;